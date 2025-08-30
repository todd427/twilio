#!/usr/bin/env python3
"""
app_toddric.py — FastAPI wrapper around ChatEngine with per-session state.

Run:
  uvicorn app_toddric:app --host 0.0.0.0 --port 8000 --workers 1

Environment (optional):
  TODDRIC_MODEL=/path/or/repo                         # default: bf16 merged path
  TODDRIC_DEVICE_MAP={"":0} or "auto"                 # default: {"":0}
  TODDRIC_ATTN=eager|sdpa                             # default: eager
  TODDRIC_ALLOW_DOMAINS=youtube.com,youtu.be          # default: youtube only
  TODDRIC_SMS_MAXNEW=60                               # SMS cap
"""
from __future__ import annotations
import os, threading, uuid
from typing import Dict, Optional, Any
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from toddric_chat import (
    ChatEngine, EngineConfig, URLSettings, ReplySettings, DecodeSettings,
    build_system_text, first_word, infer_identity_from_system, make_identity_regex
)

# ---- Config from env ----
MODEL_PATH = os.environ.get("TODDRIC_MODEL", "/home/todd/training/ckpts/toddric-3b-merged-v3")
DEVICE_MAP = os.environ.get("TODDRIC_DEVICE_MAP", None)
if DEVICE_MAP:
    try:
        # Allow JSON-like dict in env
        DEVICE_MAP = eval(DEVICE_MAP, {"__builtins__": {}})
    except Exception:
        pass
else:
    DEVICE_MAP = {"": 0}
ATTN = os.environ.get("TODDRIC_ATTN", "eager")
ALLOW_DOMAINS = [d for d in os.environ.get("TODDRIC_ALLOW_DOMAINS", "youtube.com,youtu.be").split(",") if d]
SMS_MAXNEW = int(os.environ.get("TODDRIC_SMS_MAXNEW", "60"))

# ---- Engine (single global) ----
base_cfg = EngineConfig(
    model=MODEL_PATH,
    device_map=DEVICE_MAP,
    attn_implementation=ATTN,
    trust_remote_code=True,
    bits=None,  # use bf16 path by default; set to 4/8 if you really want bnb
    system="You are a helpful assistant. Speak plainly. No HTML/markdown. Be concise.",
    persona={}, name=None,
    you_label="You", assistant_label=None,
    max_turns=1,
    url=URLSettings(
        no_urls=False, allow_domains=ALLOW_DOMAINS,
        validate_urls=True, url_timeout=2.0, link_style="inline",
    ),
    reply=ReplySettings(
        answer_first=True, no_greetings=True, no_praise=True, no_emojis=True,
        strip_identity="*",
    ),
    decode=DecodeSettings(
        max_new_tokens=160, temperature=0.2, top_p=0.95, top_k=None, repetition_penalty=1.05,
    ),
    stops=None,
)
engine = ChatEngine(base_cfg)

# ---- Session state ----
class SessionState(BaseModel):
    session_id: str
    system_text: str
    asst_label: str
    you_label: str = "You"
    id_regex_str: Optional[str] = "*"
    messages: list = Field(default_factory=list)

sessions: Dict[str, SessionState] = {}
sess_lock = threading.Lock()

# ---- FastAPI app ----
app = FastAPI(title="Toddric Chat API", version="2.0.0")

# ---- Schemas ----
class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    persona: Optional[dict] = None
    name: Optional[str] = None
    style: Optional[str] = None           # e.g. "sms_short"
    instruction: Optional[str] = None
    max_new_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None

class ChatResponse(BaseModel):
    session_id: str
    label: str
    text: str
    latency_s: float

class ResetRequest(BaseModel):
    session_id: str

class SystemRequest(BaseModel):
    session_id: str
    system: str

class PersonaRequest(BaseModel):
    session_id: str
    persona: dict
    name: Optional[str] = None

# ---- Helpers ----
def _get_or_create_session(session_id: Optional[str], persona: Optional[dict], name: Optional[str]) -> SessionState:
    with sess_lock:
        sid = session_id or uuid.uuid4().hex
        st = sessions.get(sid)
        if st: return st
        # Build system + label for this session
        sys_text = build_system_text(engine.cfg.system, persona or {}, name)
        nm = name or (persona or {}).get("name")
        asst_label = engine.asst_label if not nm else first_word(nm, "Assistant")
        ident = "*"  # default: strip any identity opener
        st = SessionState(session_id=sid, system_text=sys_text, asst_label=asst_label, you_label=engine.you_label, id_regex_str=ident, messages=[])
        sessions[sid] = st
        return st

def _apply_session(st: SessionState):
    # Swap engine state
    engine.messages = list(st.messages)
    engine.system_text = st.system_text
    engine.asst_label = st.asst_label
    engine.you_label = st.you_label
    engine.id_regex = make_identity_regex(st.id_regex_str) if st.id_regex_str else engine.id_regex

def _save_session(st: SessionState):
    st.messages = list(engine.messages)
    st.system_text = engine.system_text
    st.asst_label = engine.asst_label

# ---- Routes ----
@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_PATH, "sessions": len(sessions)}

@app.get("/whoami")
def whoami():
    try:
        p = next(engine.mdl.parameters())
        dtype = str(p.dtype); device = str(p.device)
    except StopIteration:
        dtype, device = "unknown", "unknown"
    qconf = getattr(engine.mdl, "quantization_config", None)
    if qconf:
        quant = "4bit" if getattr(qconf, "load_in_4bit", False) else ("8bit" if getattr(qconf, "load_in_8bit", False) else qconf.__class__.__name__)
    else:
        quant = None
    return {
        "model": getattr(engine.mdl, "name_or_path", None) or engine.cfg.model,
        "assistant_label": engine.asst_label,
        "device_map": engine.cfg.device_map,
        "device": device,
        "dtype": dtype,
        "quantization": quant,
        "tokenizer": getattr(engine.tok, "name_or_path", None),
        "eos_token_id": engine.tok.eos_token_id,
        "pad_token_id": engine.tok.pad_token_id,
    }

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    st = _get_or_create_session(req.session_id, req.persona, req.name)
    with sess_lock:
        _apply_session(st)
    t0 = os.times()[4]
    try:
        if (req.style or "").lower() == "sms_short":
            cap = max(24, min(SMS_MAXNEW, req.max_new_tokens or SMS_MAXNEW))
            text = engine.chat_fast_sms(req.message, max_new=cap, sentences=2, instruction=req.instruction)
        else:
            # allow lightweight overrides
            if req.max_new_tokens is not None: engine.cfg.decode.max_new_tokens = int(req.max_new_tokens)
            if req.temperature    is not None: engine.cfg.decode.temperature    = float(req.temperature)
            if req.top_p          is not None: engine.cfg.decode.top_p          = float(req.top_p)
            if req.top_k          is not None: engine.cfg.decode.top_k          = int(req.top_k)
            text = engine.chat(req.message)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"generation_error: {e}")
    finally:
        with sess_lock:
            _save_session(st)
    latency = round(os.times()[4] - t0, 3)
    return ChatResponse(session_id=st.session_id, label=st.asst_label, text=text, latency_s=latency)

@app.post("/reset")
def reset(req: ResetRequest):
    st = sessions.get(req.session_id)
    if not st: raise HTTPException(status_code=404, detail="unknown session_id")
    st.messages = []
    return {"status": "ok", "session_id": req.session_id}

@app.post("/system")
def set_system(req: SystemRequest):
    st = sessions.get(req.session_id)
    if not st: raise HTTPException(status_code=404, detail="unknown session_id")
    st.system_text = req.system.strip()
    st.id_regex_str = "*"  # keep stripping any identity opener
    st.messages = []
    return {"status": "ok", "session_id": st.session_id}

@app.post("/persona")
def set_persona(req: PersonaRequest):
    st = sessions.get(req.session_id)
    if not st: raise HTTPException(status_code=404, detail="unknown session_id")
    sys_text = build_system_text(engine.cfg.system, req.persona or {}, req.name)
    nm = req.name or (req.persona or {}).get("name")
    st.system_text = sys_text
    st.asst_label = engine.asst_label if not nm else first_word(nm, "Assistant")
    st.id_regex_str = "*"
    st.messages = []
    return {"status": "ok", "session_id": st.session_id, "label": st.asst_label}
