import os, re, logging, time, statistics, torch
from collections import deque, defaultdict
from datetime import datetime
from zoneinfo import ZoneInfo
from fastapi import FastAPI, HTTPException, Query, Request
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM

# ---------- Config ----------
MODEL = os.getenv("TODDRIC_MODEL", "/home/todd/training/ckpts/toddric-1_5b-merged-v1")
DEFAULT_TZ = os.getenv("TODDRIC_DEFAULT_TZ", "UTC")  # e.g. "Europe/Dublin"

# ---------- Logging ----------
logger = logging.getLogger("toddric")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# ---------- GPU setup ----------
torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision("high")
device = torch.device("cuda:0")
dtype = torch.bfloat16  # fallback to float16 if needed

tok = AutoTokenizer.from_pretrained(MODEL, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL, torch_dtype=dtype, low_cpu_mem_usage=True
).to(device).eval()
try:
    model.config.attn_implementation = "sdpa"
except Exception:
    pass
if getattr(tok, "pad_token_id", None) is None:
    tok.pad_token_id = tok.eos_token_id

# ---------- In-memory per-session timezone store ----------
tz_store: dict[str, str] = {}

def valid_tz(tz: str) -> bool:
    try:
        ZoneInfo(tz)
        return True
    except Exception:
        return False

def resolve_tz(session_id: str | None, req_tz: str | None) -> str:
    if req_tz and valid_tz(req_tz):
        return req_tz
    if session_id:
        tz = tz_store.get(session_id)
        if tz and valid_tz(tz):
            return tz
    return DEFAULT_TZ

# ---------- Simple metrics ----------
METRIC_WINDOW = 200
all_requests_total = 0
chat_requests_total = 0
chat_errors_total = 0
chat_lat_ms = deque(maxlen=METRIC_WINDOW)  # rolling window latencies (ms)
server_started_at = datetime.utcnow().isoformat() + "Z"

# Per-channel counters
channels = ("sms", "wa", "unknown")
chat_by_channel = defaultdict(int)      # requests per channel
errors_by_channel = defaultdict(int)    # errors per channel

def get_percentiles(values):
    if not values:
        return 0.0, 0.0, 0.0, 0.0
    v = sorted(values)
    def pct(p):
        k = (len(v) - 1) * (p / 100.0)
        f = int(k); c = min(f + 1, len(v) - 1)
        if f == c: return float(v[f])
        return float(v[f] * (c - k) + v[c] * (k - f))
    avg = statistics.mean(v)
    return avg, pct(50), pct(95), pct(99)

def metrics_json():
    avg, p50, p95, p99 = get_percentiles(list(chat_lat_ms))
    return {
        "server": "toddric@uvicorn",
        "started_at": server_started_at,
        "requests_total": all_requests_total,
        "chat_requests_total": chat_requests_total,
        "chat_errors_total": chat_errors_total,
        "chat_latency_ms": {
            "avg": round(avg, 2),
            "p50": round(p50, 2),
            "p95": round(p95, 2),
            "p99": round(p99, 2),
            "window": METRIC_WINDOW,
            "samples": len(chat_lat_ms),
        },
        "channels": {
            ch: {"chat_requests_total": chat_by_channel.get(ch, 0),
                 "chat_errors_total": errors_by_channel.get(ch, 0)}
            for ch in channels
        }
    }

def metrics_prom():
    m = metrics_json()
    lines = [
        "# HELP toddric_requests_total Total HTTP requests.",
        "# TYPE toddric_requests_total counter",
        f"toddric_requests_total {m['requests_total']}",
        "# HELP toddric_chat_requests_total Total /chat requests.",
        "# TYPE toddric_chat_requests_total counter",
        f"toddric_chat_requests_total {m['chat_requests_total']}",
        "# HELP toddric_chat_errors_total Total /chat errors.",
        "# TYPE toddric_chat_errors_total counter",
        f"toddric_chat_errors_total {m['chat_errors_total']}",
        "# HELP toddric_chat_latency_ms Rolling latency statistics.",
        "# TYPE toddric_chat_latency_ms gauge",
        f"toddric_chat_latency_ms{{quantile=\"avg\"}} {m['chat_latency_ms']['avg']}",
        f"toddric_chat_latency_ms{{quantile=\"p50\"}} {m['chat_latency_ms']['p50']}",
        f"toddric_chat_latency_ms{{quantile=\"p95\"}} {m['chat_latency_ms']['p95']}",
        f"toddric_chat_latency_ms{{quantile=\"p99\"}} {m['chat_latency_ms']['p99']}",
        f'toddric_info{{server="{m["server"]}",started_at="{m["started_at"]}"}} 1',
    ]
    # Per-channel labeled counters
    for ch in channels:
        lines.append("# HELP toddric_channel_chat_requests_total /chat requests by channel.")
        lines.append("# TYPE toddric_channel_chat_requests_total counter")
        lines.append(f'toddric_channel_chat_requests_total{{channel="{ch}"}} {chat_by_channel.get(ch,0)}')
        lines.append("# HELP toddric_channel_chat_errors_total /chat errors by channel.")
        lines.append("# TYPE toddric_channel_chat_errors_total counter")
        lines.append(f'toddric_channel_chat_errors_total{{channel="{ch}"}} {errors_by_channel.get(ch,0)}')
    return "\n".join(lines) + "\n"

# ---------- Schemas ----------
class ChatReq(BaseModel):
    message: str
    max_new_tokens: int = 48
    temperature: float = 0.3
    instruction: str | None = None
    session_id: str | None = None  # Twilio Function should pass event.From
    timezone: str | None = None
    channel: str | None = None     # "sms" | "wa" (added)

class TzReq(BaseModel):
    session_id: str
    tz: str

app = FastAPI()

@app.middleware("http")
async def count_all_requests(request: Request, call_next):
    global all_requests_total
    all_requests_total += 1
    return await call_next(request)

@app.get("/whoami")
def whoami():
    return {"model": "toddric-1_5b-merged-v1@uvicorn"}

@app.get("/healthz")
def healthz():
    return {"status": "ok", "time": datetime.utcnow().isoformat() + "Z"}

@app.get("/metrics")
def metrics(format: str = Query("json", pattern="^(json|prom)$")):
    if format == "prom":
        from fastapi.responses import PlainTextResponse
        return PlainTextResponse(metrics_prom(), media_type="text/plain; version=0.0.4")
    return metrics_json()

@app.get("/tz")
def get_tz(session_id: str = Query(..., description="sender/session id")):
    tz = tz_store.get(session_id, DEFAULT_TZ)
    return {"session_id": session_id, "tz": tz}

@app.post("/tz")
def set_tz(body: TzReq):
    if not valid_tz(body.tz):
        raise HTTPException(400, f"invalid_tz: {body.tz}")
    tz_store[body.session_id] = body.tz
    logger.info("TZ set via /tz: session_id=%s tz=%s", body.session_id, body.tz)
    return {"ok": True, "session_id": body.session_id, "tz": body.tz}

# ---------- Simple GPU concurrency gate ----------
import asyncio
gate = asyncio.Semaphore(1)

def norm_channel(ch: str | None) -> str:
    ch = (ch or "").lower().strip()
    if ch in ("sms", "wa"): return ch
    return "unknown"

@app.post("/chat")
async def chat(req: ChatReq):
    global chat_requests_total, chat_errors_total
    chat_requests_total += 1
    t0 = time.perf_counter()

    msg = (req.message or "").strip()
    if not msg:
        chat_errors_total += 1
        errors_by_channel[norm_channel(req.channel)] += 1
        raise HTTPException(400, "empty message")

    chan = norm_channel(req.channel)
    chat_by_channel[chan] += 1

    logger.info("CHAT in: channel=%s session_id=%s supplied_tz=%s msg=%r", chan, req.session_id, req.timezone, msg)

    msg_norm = msg.replace("\u00A0", " ")

    # --- Text metrics snapshot for SMS/WA ---
    if msg_norm.upper() in {"STATS", "METRICS"}:
        m = metrics_json()
        lat = m["chat_latency_ms"]
        ch_counts = m["channels"]
        # compact single-/two-segment safe text
        txt = (
            f"Reqs:{m['requests_total']} Chat:{m['chat_requests_total']} "
            f"Err:{m['chat_errors_total']} | "
            f"ms avg/p50/p95/p99:{lat['avg']}/{lat['p50']}/{lat['p95']}/{lat['p99']} "
            f"(n={lat['samples']}) | "
            f"SMS:{ch_counts['sms']['chat_requests_total']}/{ch_counts['sms']['chat_errors_total']} "
            f"WA:{ch_counts['wa']['chat_requests_total']}/{ch_counts['wa']['chat_errors_total']}"
        )
        dt = (time.perf_counter() - t0) * 1000
        chat_lat_ms.append(dt)
        return {"text": txt}

    # --- TZ command: "TZ Europe/Dublin" (case-insensitive) ---
    if msg_norm.upper().startswith("TZ "):
        if not req.session_id:
            chat_errors_total += 1
            errors_by_channel[chan] += 1
            raise HTTPException(400, "session_id required to set timezone")
        tz_candidate = msg_norm[3:].strip().strip(".,;:!\"'")
        if not tz_candidate:
            return {"text": "Usage: TZ <IANA/Zone>, e.g., TZ Europe/Dublin"}
        if not valid_tz(tz_candidate):
            return {"text": f"Invalid timezone: {tz_candidate}"}
        tz_store[req.session_id] = tz_candidate
        logger.info("TZ set via chat: session_id=%s tz=%s", req.session_id, tz_candidate)
        now = datetime.now(ZoneInfo(tz_candidate)).strftime("%H:%M")
        dt = (time.perf_counter() - t0) * 1000
        chat_lat_ms.append(dt)
        return {"text": f"Timezone set to {tz_candidate}. Local time is {now}."}

    # --- Clock hook ---
    lower = msg_norm.lower()
    if ("time" in lower and "?" in lower) or lower in {"time", "what time is it", "current time"}:
        tz = resolve_tz(req.session_id, req.timezone)
        now = datetime.now(ZoneInfo(tz))
        logger.info("Clock hook: session_id=%s resolved_tz=%s", req.session_id, tz)
        dt = (time.perf_counter() - t0) * 1000
        chat_lat_ms.append(dt)
        return {"text": f"The current time in {tz} is {now.strftime('%H:%M')}."}

    # --- Normal generation ---
    sys = req.instruction or (
        "You are Toddric — pragmatic, wry, helpful. "
        "Constraints: 1–2 sentences, no fluff, no markdown, SMS-length."
    )
    messages = [{"role": "system", "content": sys}, {"role": "user", "content": msg}]
    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tok(prompt, return_tensors="pt").to(device)

    try:
        async with gate:
            temp = float(req.temperature if req.temperature is not None else 0.3)
            do_sample = temp > 0.0
            gen_kwargs = {
                "max_new_tokens": min(128, int(req.max_new_tokens)),
                "repetition_penalty": 1.05,
                "eos_token_id": tok.eos_token_id,
                "pad_token_id": tok.pad_token_id,
                "do_sample": do_sample,
            }
            if do_sample:
                gen_kwargs.update({"temperature": max(0.05, temp), "top_p": 0.9})
            with torch.inference_mode():
                out = model.generate(**inputs, **gen_kwargs)
    except Exception:
        chat_errors_total += 1
        errors_by_channel[chan] += 1
        raise

    # decode only new tokens
    input_len = inputs["input_ids"].shape[1]
    gen_ids = out[0][input_len:]
    text = tok.decode(gen_ids, skip_special_tokens=True).strip()
    text = " ".join(text.split())

    dt = (time.perf_counter() - t0) * 1000
    chat_lat_ms.append(dt)
    logger.info("CHAT out: channel=%s session_id=%s ms=%.1f chars=%d", chan, req.session_id, dt, len(text))
    return {"text": text or "OK"}
