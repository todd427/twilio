import os, re, logging, torch
from datetime import datetime
from zoneinfo import ZoneInfo
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL = os.getenv("TODDRIC_MODEL", "/models/toddric-1_5b-merged-v1")
DEFAULT_TZ = os.getenv("TODDRIC_DEFAULT_TZ", "UTC")  # e.g. "Europe/Dublin"

# ---- logging ----
logger = logging.getLogger("toddric")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# ---- GPU ----
torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision("high")
device = torch.device("cuda:0")
dtype = torch.bfloat16

tok = AutoTokenizer.from_pretrained(MODEL, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL, torch_dtype=dtype, low_cpu_mem_usage=True
).to(device).eval()
try: model.config.attn_implementation = "sdpa"
except Exception: pass
if getattr(tok, "pad_token_id", None) is None:
    tok.pad_token_id = tok.eos_token_id

# ---- per-session TZ store ----
tz_store: dict[str, str] = {}

def valid_tz(tz: str) -> bool:
    try:
        ZoneInfo(tz); return True
    except Exception:
        return False

def resolve_tz(session_id: str | None, req_tz: str | None) -> str:
    if req_tz and valid_tz(req_tz): return req_tz
    if session_id:
        tz = tz_store.get(session_id)
        if tz and valid_tz(tz): return tz
    return DEFAULT_TZ

# ---- schemas ----
class ChatReq(BaseModel):
    message: str
    max_new_tokens: int = 48
    temperature: float = 0.3
    instruction: str | None = None
    session_id: str | None = None
    timezone: str | None = None

class TzReq(BaseModel):
    session_id: str
    tz: str

app = FastAPI()

@app.get("/whoami")
def whoami(): return {"model": "toddric-1_5b-merged-v1@uvicorn"}

@app.get("/healthz")
def healthz(): return {"status": "ok", "time": datetime.utcnow().isoformat() + "Z"}

@app.get("/tz")
def get_tz(session_id: str = Query(...)): return {"session_id": session_id, "tz": tz_store.get(session_id, DEFAULT_TZ)}

@app.post("/tz")
def set_tz(body: TzReq):
    if not valid_tz(body.tz): raise HTTPException(400, f"invalid_tz: {body.tz}")
    tz_store[body.session_id] = body.tz
    logger.info("TZ set via /tz: session_id=%s tz=%s", body.session_id, body.tz)
    return {"ok": True, "session_id": body.session_id, "tz": body.tz}

# ---- generation gate ----
import asyncio
gate = asyncio.Semaphore(1)

@app.post("/chat")
async def chat(req: ChatReq):
    msg = (req.message or "").strip()
    if not msg: raise HTTPException(400, "empty message")
    logger.info("CHAT in: session_id=%s supplied_tz=%s msg=%r", req.session_id, req.timezone, msg)

    # --- TZ command: accept any whitespace (incl NBSP) after 'TZ' ---
    # Normalize NBSPs to regular spaces first
    msg_nbsp_norm = msg.replace("\u00A0", " ")
    if msg_nbsp_norm.upper().startswith("TZ "):
        if not req.session_id:
            raise HTTPException(400, "session_id required to set timezone")
        tz_candidate = msg_nbsp_norm[3:].strip().strip(".,;:!\"'")
        if not tz_candidate:
            return {"text": "Usage: TZ <IANA/Zone>, e.g., TZ Europe/Dublin"}
        if not valid_tz(tz_candidate):
            return {"text": f"Invalid timezone: {tz_candidate}"}
        tz_store[req.session_id] = tz_candidate
        logger.info("TZ set via chat: session_id=%s tz=%s", req.session_id, tz_candidate)
        now = datetime.now(ZoneInfo(tz_candidate)).strftime("%H:%M")
        return {"text": f"Timezone set to {tz_candidate}. Local time is {now}."}

    # --- Clock hook using stored/requested timezone ---
    lower = msg.lower()
    if ("time" in lower and "?" in lower) or lower in {"time", "what time is it", "current time"}:
        tz = resolve_tz(req.session_id, req.timezone)
        now = datetime.now(ZoneInfo(tz))
        logger.info("Clock hook: session_id=%s resolved_tz=%s", req.session_id, tz)
        return {"text": f"The current time in {tz} is {now.strftime('%H:%M')}."}

    # --- System contract ---
    sys = req.instruction or (
        "You are Toddric — pragmatic, wry, helpful. "
        "Constraints: 1–2 sentences, no fluff, no markdown, SMS-length."
    )
    messages = [{"role": "system", "content": sys}, {"role": "user", "content": msg}]
    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tok(prompt, return_tensors="pt").to(device)

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
        if do_sample: gen_kwargs.update({"temperature": max(0.05, temp), "top_p": 0.9})
        with torch.inference_mode():
            out = model.generate(**inputs, **gen_kwargs)

    # decode only new tokens
    input_len = inputs["input_ids"].shape[1]
    gen_ids = out[0][input_len:]
    text = tok.decode(gen_ids, skip_special_tokens=True).strip()
    text = " ".join(text.split())
    logger.info("CHAT out: session_id=%s chars=%d", req.session_id, len(text))
    return {"text": text or "OK"}
