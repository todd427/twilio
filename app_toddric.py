import os, re, logging, time, statistics, sqlite3, torch
from collections import deque, defaultdict
from datetime import datetime
from zoneinfo import ZoneInfo
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM

# ---------- Config ----------
MODEL = os.getenv("TODDRIC_MODEL", "/home/todd/training/ckpts/toddric-1_5b-merged-v1")
DEFAULT_TZ = os.getenv("TODDRIC_DEFAULT_TZ", "UTC")       # e.g. "Europe/Dublin"
DB_PATH = os.getenv("TODDRIC_DB", "./toddric.db")         # SQLite file

# ---------- Logging ----------
logger = logging.getLogger("toddric")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# ---------- DB ----------
def db():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("""
      CREATE TABLE IF NOT EXISTS user_prefs(
        session_id TEXT PRIMARY KEY,
        tz TEXT,
        updated_at TEXT
      );
    """)
    conn.execute("""
      CREATE TABLE IF NOT EXISTS chunks(
        session_id TEXT PRIMARY KEY,
        text TEXT NOT NULL,
        offset INTEGER NOT NULL DEFAULT 0,
        updated_at TEXT
      );
    """)
    conn.execute("""
      CREATE TABLE IF NOT EXISTS memory(
        session_id TEXT,
        key TEXT,
        value TEXT,
        updated_at TEXT,
        PRIMARY KEY(session_id, key)
      );
    """)
    conn.commit()
    return conn

DB = db()

def db_get_tz(session_id: str):
    cur = DB.execute("SELECT tz FROM user_prefs WHERE session_id=?", (session_id,))
    row = cur.fetchone()
    return row[0] if row else None

def db_set_tz(session_id: str, tz: str):
    now = datetime.utcnow().isoformat() + "Z"
    DB.execute(
        "INSERT INTO user_prefs(session_id, tz, updated_at) VALUES(?,?,?) "
        "ON CONFLICT(session_id) DO UPDATE SET tz=excluded.tz, updated_at=excluded.updated_at",
        (session_id, tz, now)
    )
    DB.commit()

def db_store_chunk(session_id: str, text: str):
    now = datetime.utcnow().isoformat() + "Z"
    DB.execute(
        "INSERT INTO chunks(session_id, text, offset, updated_at) VALUES(?,?,0,?) "
        "ON CONFLICT(session_id) DO UPDATE SET text=excluded.text, offset=0, updated_at=excluded.updated_at",
        (session_id, text, now)
    )
    DB.commit()

def db_next_chunk(session_id: str, max_len: int):
    cur = DB.execute("SELECT text, offset FROM chunks WHERE session_id=?", (session_id,))
    row = cur.fetchone()
    if not row:
        return "", 0, True
    text, offset = row
    if offset >= len(text):
        return "", offset, True
    # Find a nice break ≤ max_len: prefer whitespace/punct
    end = min(offset + max_len, len(text))
    if end < len(text):
        slice_ = text[offset:end]
        # try to cut at last whitespace or punctuation
        m = re.search(r"[ \t\n\r.,;:!?](?!.*[ \t\n\r.,;:!?])", slice_)
        cut = end
        if m:
            cut = offset + m.start() + 1  # include the boundary char
        else:
            # fallback: walk backwards to first space in window
            for i in range(end-1, offset, -1):
                if text[i].isspace():
                    cut = i
                    break
        end = max(cut, offset + min(40, len(text)-offset))  # guard against zero progress
    chunk = text[offset:end].strip()
    new_off = end
    done = new_off >= len(text)
    now = datetime.utcnow().isoformat() + "Z"
    DB.execute("UPDATE chunks SET offset=?, updated_at=? WHERE session_id=?", (new_off, now, session_id))
    DB.commit()
    return chunk, new_off, done

# (optional) simple memory helpers — ready to use later
def db_memory_set(session_id: str, key: str, value: str):
    now = datetime.utcnow().isoformat() + "Z"
    DB.execute(
        "INSERT INTO memory(session_id, key, value, updated_at) VALUES(?,?,?,?) "
        "ON CONFLICT(session_id, key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at",
        (session_id, key, value, now)
    )
    DB.commit()

def db_memory_get(session_id: str, key: str):
    cur = DB.execute("SELECT value FROM memory WHERE session_id=? AND key=?", (session_id, key))
    row = cur.fetchone()
    return row[0] if row else None

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

# ---------- Metrics ----------
METRIC_WINDOW = 200
all_requests_total = 0
chat_requests_total = 0
chat_errors_total = 0
chat_lat_ms = deque(maxlen=METRIC_WINDOW)
server_started_at = datetime.utcnow().isoformat() + "Z"
channels = ("sms", "wa", "unknown")
chat_by_channel = defaultdict(int)
errors_by_channel = defaultdict(int)

def get_percentiles(values):
    if not values:
        return 0.0, 0.0, 0.0, 0.0
    v = sorted(values)
    def pct(p):
        k = (len(v) - 1) * (p / 100.0)
        f = int(k); c = min(f + 1, len(v) - 1)
        if f == c: return float(v[f])
        return float(v[f] * (c - k) + v[c] * (k - f))
    import statistics
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
            "avg": round(avg, 2), "p50": round(p50, 2),
            "p95": round(p95, 2), "p99": round(p99, 2),
            "window": METRIC_WINDOW, "samples": len(chat_lat_ms),
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
    for ch in channels:
        lines.append("# TYPE toddric_channel_chat_requests_total counter")
        lines.append(f'toddric_channel_chat_requests_total{{channel="{ch}"}} {chat_by_channel.get(ch,0)}')
        lines.append("# TYPE toddric_channel_chat_errors_total counter")
        lines.append(f'toddric_channel_chat_errors_total{{channel="{ch}"}} {errors_by_channel.get(ch,0)}')
    return "\n".join(lines) + "\n"

# ---------- Schemas ----------
class ChatReq(BaseModel):
    message: str
    max_new_tokens: int = 48
    temperature: float = 0.3
    instruction: str | None = None
    session_id: str | None = None
    timezone: str | None = None
    channel: str | None = None

class TzReq(BaseModel):
    session_id: str
    tz: str

class MoreReq(BaseModel):
    session_id: str
    max_chars: int = 300

# ---------- App ----------
from fastapi import FastAPI
app = FastAPI()

@app.middleware("http")
async def count_all_requests(request: Request, call_next):
    global all_requests_total
    all_requests_total += 1
    return await call_next(request)

@app.get("/whoami")
def whoami(): return {"model": "toddric-1_5b-merged-v1@uvicorn"}

@app.get("/healthz")
def healthz(): return {"status": "ok", "time": datetime.utcnow().isoformat() + "Z"}

@app.get("/metrics")
def metrics(format: str = Query("json", pattern="^(json|prom)$")):
    return PlainTextResponse(metrics_prom(), media_type="text/plain; version=0.0.4") if format=="prom" else metrics_json()

@app.get("/tz")
def get_tz(session_id: str = Query(..., description="sender/session id")):
    return {"session_id": session_id, "tz": db_get_tz(session_id) or DEFAULT_TZ}

@app.post("/tz")
def set_tz(body: TzReq):
    if not body.session_id: raise HTTPException(400, "session_id required")
    try:
        ZoneInfo(body.tz)
    except Exception:
        raise HTTPException(400, f"invalid_tz: {body.tz}")
    db_set_tz(body.session_id, body.tz)
    logger.info("TZ set via /tz: session_id=%s tz=%s", body.session_id, body.tz)
    return {"ok": True, "session_id": body.session_id, "tz": body.tz}

@app.post("/more")
def more(body: MoreReq):
    if not body.session_id: raise HTTPException(400, "session_id required")
    max_chars = max(80, min(600, int(body.max_chars or 300)))
    chunk, pos, done = db_next_chunk(body.session_id, max_chars)
    return {"chunk": chunk, "offset": pos, "done": bool(done)}

# ---------- Helpers ----------
import asyncio
gate = asyncio.Semaphore(1)
def norm_channel(ch: str | None) -> str:
    ch = (ch or "").lower().strip()
    return ch if ch in ("sms","wa") else "unknown"

def resolve_tz(session_id: str | None, req_tz: str | None) -> str:
    if req_tz:
        try: ZoneInfo(req_tz); return req_tz
        except Exception: pass
    if session_id:
        tz = db_get_tz(session_id)
        if tz:
            try: ZoneInfo(tz); return tz
            except Exception: pass
    return DEFAULT_TZ

# ---------- Main /chat ----------
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

    # STATS/metrics via SMS
    if msg_norm.upper() in {"STATS", "METRICS"}:
        m = metrics_json(); lat = m["chat_latency_ms"]; ch = m["channels"]
        txt = (f"Reqs:{m['requests_total']} Chat:{m['chat_requests_total']} Err:{m['chat_errors_total']} | "
               f"ms avg/p50/p95/p99:{lat['avg']}/{lat['p50']}/{lat['p95']}/{lat['p99']} (n={lat['samples']}) | "
               f"SMS:{ch['sms']['chat_requests_total']}/{ch['sms']['chat_errors_total']} "
               f"WA:{ch['wa']['chat_requests_total']}/{ch['wa']['chat_errors_total']}")
        dt = (time.perf_counter()-t0)*1000; chat_lat_ms.append(dt)
        return {"text": txt}

    # TZ command: "TZ Europe/Dublin"
    if msg_norm.upper().startswith("TZ "):
        if not req.session_id:
            chat_errors_total += 1; errors_by_channel[chan] += 1
            raise HTTPException(400, "session_id required to set timezone")
        tz_candidate = msg_norm[3:].strip().strip(".,;:!\"'")
        try:
            ZoneInfo(tz_candidate)
        except Exception:
            return {"text": f"Invalid timezone: {tz_candidate}"}
        db_set_tz(req.session_id, tz_candidate)
        now = datetime.now(ZoneInfo(tz_candidate)).strftime("%H:%M")
        dt = (time.perf_counter()-t0)*1000; chat_lat_ms.append(dt)
        return {"text": f"Timezone set to {tz_candidate}. Local time is {now}."}

    # Clock hook
    lower = msg_norm.lower()
    if ("time" in lower and "?" in lower) or lower in {"time", "what time is it", "current time"}:
        tz = resolve_tz(req.session_id, req.timezone)
        now = datetime.now(ZoneInfo(tz))
        dt = (time.perf_counter()-t0)*1000; chat_lat_ms.append(dt)
        return {"text": f"The current time in {tz} is {now.strftime('%H:%M')}."}

    # Generation
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

    input_len = inputs["input_ids"].shape[1]
    gen_ids = out[0][input_len:]
    text = tok.decode(gen_ids, skip_special_tokens=True).strip()
    text = " ".join(text.split())

    # store full text for MORE pagination
    if req.session_id:
        db_store_chunk(req.session_id, text)

    dt = (time.perf_counter()-t0)*1000; chat_lat_ms.append(dt)
    return {"text": text or "OK"}
