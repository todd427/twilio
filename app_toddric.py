import os, re, logging, time, statistics, sqlite3, requests, torch
from collections import deque, defaultdict
from datetime import datetime
from zoneinfo import ZoneInfo
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM

# ---------- Config ----------
MODEL = os.getenv("TODDRIC_MODEL", "/home/todd/training/ckpts/toddric-1_5b-merged-v1")
DEFAULT_TZ = os.getenv("TODDRIC_DEFAULT_TZ", "UTC")           # e.g. "Europe/Dublin"
DB_PATH = os.getenv("TODDRIC_DB", "./toddric.db")             # SQLite file path

# ---------- Logging ----------
logger = logging.getLogger("toddric")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

def now_iso() -> str:
    return datetime.utcnow().isoformat() + "Z"

# ---------- DB Setup ----------
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

# --- user prefs (tz) ---
def db_get_tz(session_id: str):
    cur = DB.execute("SELECT tz FROM user_prefs WHERE session_id=?", (session_id,))
    row = cur.fetchone()
    return row[0] if row else None

def db_set_tz(session_id: str, tz: str):
    DB.execute(
        "INSERT INTO user_prefs(session_id, tz, updated_at) VALUES(?,?,?) "
        "ON CONFLICT(session_id) DO UPDATE SET tz=excluded.tz, updated_at=excluded.updated_at",
        (session_id, tz, now_iso())
    )
    DB.commit()

def db_delete_session(session_id: str):
    DB.execute("DELETE FROM user_prefs WHERE session_id=?", (session_id,))
    DB.execute("DELETE FROM chunks WHERE session_id=?", (session_id,))
    DB.execute("DELETE FROM memory WHERE session_id=?", (session_id,))
    DB.commit()

# --- chunks for MORE ---
def db_store_chunk(session_id: str, text: str):
    DB.execute(
        "INSERT INTO chunks(session_id, text, offset, updated_at) VALUES(?,?,0,?) "
        "ON CONFLICT(session_id) DO UPDATE SET text=excluded.text, offset=0, updated_at=excluded.updated_at",
        (session_id, text, now_iso())
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

    end = min(offset + max_len, len(text))
    if end < len(text):
        # prefer breaking at whitespace/punct within window
        slice_ = text[offset:end]
        m = re.search(r"[ \t\n\r.,;:!?](?!.*[ \t\n\r.,;:!?])", slice_)
        cut = end
        if m:
            cut = offset + m.start() + 1
        else:
            for i in range(end - 1, offset, -1):
                if text[i].isspace():
                    cut = i
                    break
        end = max(cut, offset + min(40, len(text) - offset))  # ensure progress

    chunk = text[offset:end].strip()
    new_off = end
    done = new_off >= len(text)
    DB.execute("UPDATE chunks SET offset=?, updated_at=? WHERE session_id=?", (new_off, now_iso(), session_id))
    DB.commit()
    return chunk, new_off, done

# --- memory helpers (case-insensitive keys) ---
def db_memory_set(session_id: str, key: str, value: str):
    k = (key or "").strip().lower()[:64]
    v = (value or "").strip()[:400]
    DB.execute(
        "INSERT INTO memory(session_id, key, value, updated_at) VALUES(?,?,?,?) "
        "ON CONFLICT(session_id, key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at",
        (session_id, k, v, now_iso())
    )
    DB.commit()

def db_memory_get(session_id: str, key: str):
    k = (key or "").strip()
    cur = DB.execute(
        "SELECT value FROM memory WHERE session_id=? AND lower(key)=lower(?) LIMIT 1",
        (session_id, k)
    )
    row = cur.fetchone()
    return row[0] if row else None

def db_memory_list(session_id: str):
    cur = DB.execute("SELECT key FROM memory WHERE session_id=? ORDER BY updated_at DESC", (session_id,))
    return [r[0] for r in cur.fetchall()]

def build_profile(session_id: str | None) -> str:
    if not session_id:
        return ""
    cur = DB.execute(
        "SELECT key, value FROM memory WHERE session_id=? ORDER BY updated_at DESC LIMIT 12",
        (session_id,)
    )
    items = [f"{k.strip()}: {v.strip()}" for (k, v) in cur.fetchall() if k and v]
    if not items:
        return ""
    return "Known user facts — " + "; ".join(items) + ". Use naturally if relevant."

# ---------- Weather (Open-Meteo, no key) ----------
def geocode_open_meteo(query: str):
    try:
        r = requests.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": query, "count": 1, "language": "en", "format": "json"},
            timeout=5,
        )
        j = r.json()
        if j.get("results"):
            it = j["results"][0]
            return float(it["latitude"]), float(it["longitude"]), it.get("timezone") or "UTC", it.get("name")
    except Exception:
        pass
    return None, None, None, None

def weather_open_meteo(lat: float, lon: float, tz: str):
    try:
        r = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": lat, "longitude": lon,
                "current": ["temperature_2m","precipitation"],
                "daily": ["temperature_2m_max","temperature_2m_min","precipitation_probability_max"],
                "timezone": tz,
            },
            timeout=6,
        )
        j = r.json()
        cur = j.get("current", {})
        daily = j.get("daily", {})
        t = cur.get("temperature_2m")
        p = cur.get("precipitation")
        tmax = (daily.get("temperature_2m_max") or [None])[0]
        tmin = (daily.get("temperature_2m_min") or [None])[0]
        pprob = (daily.get("precipitation_probability_max") or [None])[0]
        return t, p, tmax, tmin, pprob
    except Exception:
        return None, None, None, None, None

# ---------- GPU + Model ----------
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
server_started_at = now_iso()
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

class ResetReq(BaseModel):
    session_id: str

class MemorySet(BaseModel):
    session_id: str
    key: str
    value: str

# ---------- App ----------
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
    return {"status": "ok", "time": now_iso()}

@app.get("/metrics")
def metrics(format: str = Query("json", pattern="^(json|prom)$")):
    return PlainTextResponse(metrics_prom(), media_type="text/plain; version=0.0.4") if format=="prom" else metrics_json()

@app.get("/tz")
def get_tz(session_id: str = Query(..., description="sender/session id")):
    return {"session_id": session_id, "tz": db_get_tz(session_id) or DEFAULT_TZ}

@app.post("/tz")
def set_tz(body: TzReq):
    if not body.session_id: raise HTTPException(400, "session_id required")
    try: ZoneInfo(body.tz)
    except Exception: raise HTTPException(400, f"invalid_tz: {body.tz}")
    db_set_tz(body.session_id, body.tz)
    logger.info("TZ set via /tz: session_id=%s tz=%s", body.session_id, body.tz)
    return {"ok": True, "session_id": body.session_id, "tz": body.tz}

@app.post("/more")
def more(body: MoreReq):
    if not body.session_id: raise HTTPException(400, "session_id required")
    max_chars = max(80, min(600, int(body.max_chars or 300)))
    chunk, pos, done = db_next_chunk(body.session_id, max_chars)
    return {"chunk": chunk, "offset": pos, "done": bool(done)}

@app.post("/reset")
def reset(body: ResetReq):
    if not body.session_id: raise HTTPException(400, "session_id required")
    db_delete_session(body.session_id)
    return {"ok": True}

@app.post("/memory")
def memory_set(body: MemorySet):
    if not body.session_id or not body.key:
        raise HTTPException(400, "session_id and key required")
    db_memory_set(body.session_id, body.key, body.value)
    return {"ok": True}

@app.get("/memory")
def memory_get(session_id: str = Query(...), key: str = Query(...)):
    v = db_memory_get(session_id, key)
    if v is None:
        raise HTTPException(404, "not found")
    return {"value": v}

@app.get("/memory/list")
def memory_list(session_id: str = Query(...)):
    return {"keys": db_memory_list(session_id)}

@app.get("/weather")
def weather(loc: str = Query(..., description="city or place"), session_id: str | None = None):
    lat, lon, tz, name = geocode_open_meteo(loc)
    if lat is None:
        raise HTTPException(404, f"Could not locate '{loc}'")
    t, p, tmax, tmin, pprob = weather_open_meteo(lat, lon, tz)
    if t is None:
        raise HTTPException(503, "Weather unavailable")
    return {
        "place": name or loc,
        "tz": tz,
        "current": {"temp_c": t, "precip_mm": p},
        "today": {"tmax_c": tmax, "tmin_c": tmin, "precip_prob_pct": pprob},
    }

# ---------- Helpers ----------
import asyncio
gate = asyncio.Semaphore(1)

def norm_channel(ch: str | None) -> str:
    ch = (ch or "").lower().strip()
    return ch if ch in ("sms","wa") else "unknown"

def resolve_tz_for_req(session_id: str | None, req_tz: str | None) -> str:
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

    # First-contact welcome (persist a default tz row if totally new)
    first_contact = False
    if req.session_id:
        cur = DB.execute("SELECT 1 FROM user_prefs WHERE session_id=?", (req.session_id,))
        if not cur.fetchone():
            first_contact = True
            db_set_tz(req.session_id, DEFAULT_TZ)

    msg_norm = msg.replace("\u00A0", " ")

    # ----- Commands -----

    # STATS via chat
    if msg_norm.upper() in {"STATS", "METRICS"}:
        m = metrics_json(); lat = m["chat_latency_ms"]; ch = m["channels"]
        txt = (f"Reqs:{m['requests_total']} Chat:{m['chat_requests_total']} Err:{m['chat_errors_total']} | "
               f"ms avg/p50/p95/p99:{lat['avg']}/{lat['p50']}/{lat['p95']}/{lat['p99']} (n={lat['samples']}) | "
               f"SMS:{ch['sms']['chat_requests_total']}/{ch['sms']['chat_errors_total']} "
               f"WA:{ch['wa']['chat_requests_total']}/{ch['wa']['chat_errors_total']}")
        dt = (time.perf_counter()-t0)*1000; chat_lat_ms.append(dt)
        return {"text": txt}

    # TZ command
    if msg_norm.upper().startswith("TZ "):
        if not req.session_id:
            chat_errors_total += 1; errors_by_channel[chan] += 1
            raise HTTPException(400, "session_id required to set timezone")
        tz_candidate = msg_norm[3:].strip().strip(".,;:!\"'")
        try: ZoneInfo(tz_candidate)
        except Exception: return {"text": f"Invalid timezone: {tz_candidate}"}
        db_set_tz(req.session_id, tz_candidate)
        now_local = datetime.now(ZoneInfo(tz_candidate)).strftime("%H:%M")
        dt = (time.perf_counter()-t0)*1000; chat_lat_ms.append(dt)
        return {"text": f"Timezone set to {tz_candidate}. Local time is {now_local}."}

    # Clock hook
    lower = msg_norm.lower()
    if ("time" in lower and "?" in lower) or lower in {"time", "what time is it", "current time"}:
        tz = resolve_tz_for_req(req.session_id, req.timezone)
        now_local = datetime.now(ZoneInfo(tz)).strftime("%H:%M")
        dt = (time.perf_counter()-t0)*1000; chat_lat_ms.append(dt)
        return {"text": f"The current time in {tz} is {now_local}."}

    # WEATHER command: "/weather Dublin" or "weather Dublin"
    m_w = re.match(r"^\/?\s*weather(?:\s+(.+))?$", msg_norm, flags=re.I)
    if m_w:
        place = (m_w.group(1) or "").strip()
        if not place:
            return {"text": "Usage: WEATHER <city>, e.g., WEATHER Dublin"}
        lat, lon, wtz, name = geocode_open_meteo(place)
        if lat is None:
            return {"text": f"Couldn’t find '{place}'."}
        t, p, tmax, tmin, pprob = weather_open_meteo(lat, lon, wtz)
        if t is None:
            return {"text": "Weather service temporarily unavailable."}
        txt = f"{name}: {t:.0f}°C now, {tmin:.0f}–{tmax:.0f}°C today, rain {pprob}%."
        dt = (time.perf_counter()-t0)*1000; chat_lat_ms.append(dt)
        return {"text": txt}

    # REMEMBER key: value OR key=value OR key value
    m_mem = re.match(r"^remember\s+(.+)$", msg_norm, flags=re.I)
    if m_mem and req.session_id:
        rest = m_mem.group(1).strip()
        if ":" in rest or "=" in rest:
            parts = re.split(r"[:=]", rest, 1)
            k = parts[0].strip().lower()[:64]
            v = parts[1].strip()[:400] if len(parts) > 1 else ""
        else:
            parts = rest.split(None, 1)
            k = (parts[0].strip().lower()[:64]) if parts else ""
            v = (parts[1].strip()[:400]) if len(parts) > 1 else ""
        if not k or not v:
            return {"text": "Usage: REMEMBER key: value"}
        db_memory_set(req.session_id, k, v)
        dt = (time.perf_counter()-t0)*1000; chat_lat_ms.append(dt)
        return {"text": f"Saved {k}: {v[:120]}"}

    # RECALL key
    m_rec = re.match(r"^recall\s+(.+)$", msg_norm, flags=re.I)
    if m_rec and req.session_id:
        k = m_rec.group(1).strip()
        v = db_memory_get(req.session_id, k)
        dt = (time.perf_counter()-t0)*1000; chat_lat_ms.append(dt)
        return {"text": (v if v else f"I don’t have '{k}'.")}

    # FORGET key
    m_fgt = re.match(r"^forget\s+(.+)$", msg_norm, flags=re.I)
    if m_fgt and req.session_id:
        k = m_fgt.group(1).strip()
        DB.execute("DELETE FROM memory WHERE session_id=? AND lower(key)=lower(?)", (req.session_id, k))
        DB.commit()
        dt = (time.perf_counter()-t0)*1000; chat_lat_ms.append(dt)
        return {"text": f"Forgot {k}."}

    # LIST memory keys
    if msg_norm.upper() in {"LIST", "MEMORY", "LIST MEMORY"} and req.session_id:
        keys = db_memory_list(req.session_id)
        dt = (time.perf_counter()-t0)*1000; chat_lat_ms.append(dt)
        if not keys:
            return {"text": "I don’t have anything stored yet."}
        return {"text": "Stored keys: " + ", ".join(keys[:20])}

    # If it's the first message ever, greet and exit early
    if first_contact:
        dt = (time.perf_counter()-t0)*1000; chat_lat_ms.append(dt)
        return {"text": "Welcome to Toddric! 👋 Ask me anything in 1–2 sentences. Text HELP for House Rules."}

    # ----- Normal generation -----
    profile = build_profile(req.session_id)
    sys = req.instruction or (
        "You are Toddric — pragmatic, wry, helpful. "
        "Constraints: 1–2 sentences, no fluff, no markdown, SMS-length."
    )
    if profile:
        sys += " " + profile

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

    # Store full text for MORE pagination
    if req.session_id:
        db_store_chunk(req.session_id, text)

    dt = (time.perf_counter()-t0)*1000; chat_lat_ms.append(dt)
    logger.info("CHAT out: channel=%s session_id=%s ms=%.1f chars=%d", chan, req.session_id, dt, len(text))
    return {"text": text or "OK"}

# --- Simple JSON generator for internal callers (voice bot, etc.) ---
from typing import Optional

class GenerateReq(BaseModel):
    text: str
    session_id: Optional[str] = None
    temperature: float = 0.3
    max_new_tokens: int = 64
    timezone: Optional[str] = None

@app.post("/api/generate")
async def api_generate(body: GenerateReq):
    # Reuse the existing /chat pipeline to keep behavior consistent
    req = ChatReq(
        message=body.text,
        max_new_tokens=body.max_new_tokens,
        temperature=body.temperature,
        instruction=None,
        session_id=body.session_id,
        timezone=body.timezone,
        channel="api",
    )
    res = await chat(req)  # chat() already returns {"text": "..."}
    return {"reply": res.get("text", "").strip()}
