# app_toddric.py
# FastAPI wrapper around ChatEngine with per-session state.
# Run: uvicorn app_toddric:app --host 0.0.0.0 --port 8000 --workers 1
# pip install fastapi uvicorn pydantic transformers accelerate bitsandbytes sentencepiece

from __future__ import annotations
import os, threading, uuid
from typing import Dict, Optional, Any
from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field

from toddric_chat import (
    ChatEngine, EngineConfig, URLSettings, ReplySettings, DecodeSettings,
    load_persona, build_system_text, first_word, infer_identity_from_system,
    make_identity_regex
)

# ---- Config (env overrides optional) ----
MODEL_PATH = os.environ.get("TODDRIC_MODEL", "/home/todd/training/ckpts/toddric-3b-merged-v3-bnb4")
ALLOW_DOMAINS = os.environ.get("TODDRIC_ALLOW_DOMAINS", "youtube.com,youtu.be").split(",")

# Base engine (loads model once)
base_cfg = EngineConfig(
    model=MODEL_PATH,
    device_map="auto",
    trust_remote_code=True,
    bits=None,  # model already quantized? leave None to avoid warnings
    system="You are a helpful assistant. Speak plainly. No HTML/markdown. Be concise. Do not repeat or quote the user's message; answer directly.",
    persona={}, name=None,
    you_label="You", assistant_label=None,
    max_turns=1,
    url=URLSettings(
        no_urls=False, allow_domains=ALLOW_DOMAINS,
        validate_urls=True, url_timeout=2.5, link_style="inline",
    ),
    reply=ReplySettings(
        answer_first=True, no_greetings=True, no_praise=True, no_emojis=True,
        strip_identity="*",
    ),
    decode=DecodeSettings(
        max_new_tokens=160,   # more room so it continues after any echo
        temperature=0.2,      # tiny randomness to break deterministic echo loops
        top_p=0.95,
        top_k=None,
        repetition_penalty=1.05,
    ),
    stops=None,
)
engine = ChatEngine(base_cfg)  # loads model/tokenizer once

# ---- Simple per-session state (messages, persona/system, label) ----
class SessionState(BaseModel):
    session_id: str
    system_text: str
    asst_label: str
    you_label: str = "You"
    id_regex_str: Optional[str] = None  # store pattern string for rebuild if needed
    messages: list = Field(default_factory=list)

    def rebuild_regex(self):
        # lazily rebuild regex from stored pattern if needed (not strictly necessary here)
        return make_identity_regex("*") if self.id_regex_str == "*" else make_identity_regex(self.id_regex_str)

sessions: Dict[str, SessionState] = {}
sess_lock = threading.Lock()

# ---- FastAPI app ----
app = FastAPI(title="Toddric Chat API", version="1.0.0")

# ---- Schemas ----
class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    persona: Optional[dict] = None  # inline persona JSON (optional)
    name: Optional[str] = None      # optional override for persona.name

class ChatResponse(BaseModel):
    session_id: str
    label: str
    text: str

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
        if st:
            return st
        # Build system + label for this session
        sys_text = build_system_text(engine.cfg.system, persona or {}, name)
        nm = name or (persona or {}).get("name")
        asst_label = engine.asst_label if not nm else first_word(nm, "Assistant")
        # Choose identity strip rule: use "*" to remove any opener by default
        ident = "*" if engine.cfg.reply.strip_identity in (None, "*") else (nm or infer_identity_from_system(sys_text))
        st = SessionState(
            session_id=sid,
            system_text=sys_text,
            asst_label=asst_label,
            you_label=engine.you_label,
            id_regex_str=ident if isinstance(ident, str) else "*",
            messages=[]
        )
        sessions[sid] = st
        return st

def _run_chat_in_session(st: SessionState, message: str) -> str:
    """Temporarily swap engine state with session state, run chat, store back."""
    # Backup engine state
    with sess_lock:
        old_msgs = engine.messages
        old_sys = engine.system_text
        old_lbl = engine.asst_label
        old_you = engine.you_label
        old_regex = engine.id_regex

        # Apply session state
        engine.messages = list(st.messages)
        engine.system_text = st.system_text
        engine.asst_label = st.asst_label
        engine.you_label = st.you_label
        engine.id_regex = make_identity_regex(st.id_regex_str) if st.id_regex_str else engine.id_regex

    # Do generation off the event loop thread
    def _do():
        return engine.chat(message)

    text = None
    try:
        text = engine.chat(message)  # engine.chat already runs inference; safe here
    except Exception as e:
        # Restore on failure too
        with sess_lock:
            engine.messages = old_msgs
            engine.system_text = old_sys
            engine.asst_label = old_lbl
            engine.you_label = old_you
            engine.id_regex = old_regex
        raise e

    # Save back & restore engine globals
    with sess_lock:
        st.messages = list(engine.messages)
        st.system_text = engine.system_text
        st.asst_label = engine.asst_label
        # Restore engine to original state
        engine.messages = old_msgs
        engine.system_text = old_sys
        engine.asst_label = old_lbl
        engine.you_label = old_you
        engine.id_regex = old_regex

    return text

# ---- Routes ----
@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_PATH, "sessions": len(sessions)}

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    st = _get_or_create_session(req.session_id, req.persona, req.name)
    try:
        # run in threadpool so event loop isn't blocked
        text = engine.chat(req.message) if sess_lock.locked() else _run_chat_in_session(st, req.message)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"generation_error: {e}")
    return ChatResponse(session_id=st.session_id, label=st.asst_label, text=text)

@app.post("/reset")
def reset(req: ResetRequest):
    st = sessions.get(req.session_id)
    if not st:
        raise HTTPException(status_code=404, detail="unknown session_id")
    st.messages = []
    return {"status": "ok", "session_id": req.session_id}

@app.post("/system")
def set_system(req: SystemRequest):
    st = sessions.get(req.session_id)
    if not st:
        raise HTTPException(status_code=404, detail="unknown session_id")
    st.system_text = req.system.strip()
    # re-infer identity if we were inferring
    if st.id_regex_str is None:
        inferred = infer_identity_from_system(st.system_text)
        st.id_regex_str = inferred or "*"
    st.messages = []
    return {"status": "ok", "session_id": st.session_id}

@app.get("/whoami")
def whoami():
    # Basic runtime/model diagnostics
    try:
        p = next(engine.mdl.parameters())
        dtype = str(p.dtype)
        device = str(p.device)
    except StopIteration:
        dtype, device = "unknown", "unknown"

    qconf = getattr(engine.mdl, "quantization_config", None)
    if qconf:
        if getattr(qconf, "load_in_4bit", False):
            quant = "4bit"
        elif getattr(qconf, "load_in_8bit", False):
            quant = "8bit"
        else:
            quant = qconf.__class__.__name__
    else:
        quant = None

    info = {
        "model": getattr(engine.mdl, "name_or_path", None) or engine.cfg.model,
        "assistant_label": engine.asst_label,
        "device_map": engine.cfg.device_map,
        "device": device,
        "dtype": dtype,
        "quantization": quant,
        "tokenizer": getattr(engine.tok, "name_or_path", None),
        "eos_token_id": engine.tok.eos_token_id,
        "pad_token_id": engine.tok.pad_token_id,
        "decode": {
            "max_new_tokens": engine.cfg.decode.max_new_tokens,
            "temperature": engine.cfg.decode.temperature,
            "top_p": engine.cfg.decode.top_p,
            "top_k": engine.cfg.decode.top_k,
            "repetition_penalty": engine.cfg.decode.repetition_penalty,
        },
        "url_policy": {
            "allow_domains": engine.cfg.url.allow_domains,
            "validate_urls": engine.cfg.url.validate_urls,
            "link_style": engine.cfg.url.link_style,
        },
        "stops_count": len(engine.stops),
        "sessions": len(sessions),
    }
    return info

@app.post("/persona")
def set_persona(req: PersonaRequest):
    st = sessions.get(req.session_id)
    if not st:
        raise HTTPException(status_code=404, detail="unknown session_id")
    sys_text = build_system_text(engine.cfg.system, req.persona or {}, req.name)
    nm = req.name or (req.persona or {}).get("name")
    st.system_text = sys_text
    st.asst_label = engine.asst_label if not nm else first_word(nm, "Assistant")
    st.id_regex_str = "*"  # default to strip any identity opener
    st.messages = []
    return {"status": "ok", "session_id": st.session_id, "label": st.asst_label}

@app.delete("/session/{session_id}")
def drop_session(session_id: str):
    with sess_lock:
        if session_id in sessions:
            del sessions[session_id]
            return {"status": "ok", "session_id": session_id}
    raise HTTPException(status_code=404, detail="unknown session_id")
