#!/usr/bin/env python3
"""
Toddric Players — FastAPI + LLaMA backend with live web UI.

Run:
  uvicorn app_players:app --reload --port 8020
Then open http://localhost:8020/
"""

import os, re, json, torch, threading
from typing import Optional, Dict, Any, AsyncGenerator
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    TextIteratorStreamer, BitsAndBytesConfig
)

# ---------- Quantization helpers ----------
def build_bnb(bits: Optional[int]):
    if not bits:
        return None
    if bits == 4:
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        )
    if bits == 8:
        return BitsAndBytesConfig(load_in_8bit=True)
    raise ValueError("--bits must be 4, 8, or omitted")

def _first_word(name: Optional[str], default="Assistant") -> str:
    if not name:
        return default
    return re.sub(r"[^A-Za-z0-9_.\\-]", "", name.strip().split()[0]) or default


# ---------- Player loader ----------
def load_player(path_or_name: str, base_dir: str = "players") -> Dict[str, Any]:
    """Load a persona ('player') JSON either by path or short name."""
    if not path_or_name:
        return {}
    if os.path.isfile(path_or_name):
        path = path_or_name
    else:
        guess = os.path.join(base_dir, f"{path_or_name}.json")
        path = guess if os.path.isfile(guess) else path_or_name
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"⚠️  Could not load player '{path_or_name}': {e}")
        return {}


# ---------- Model + tokenizer setup ----------
MODEL_NAME = os.getenv("TODDRIC_MODEL", "meta-llama/Llama-3.1-8B-Instruct")
PLAYER_PATH = os.getenv("TODDRIC_PLAYER", "players/kermit.json")
BITS = int(os.getenv("TODDRIC_BITS", "4"))
TRUST_REMOTE = True

print(f"🎭 Loading model: {MODEL_NAME}")
bnb = build_bnb(BITS)
tok = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=TRUST_REMOTE)
tok.padding_side = "left"
if tok.pad_token_id is None and tok.eos_token_id is not None:
    tok.pad_token = tok.eos_token

mdl = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    trust_remote_code=TRUST_REMOTE,
    quantization_config=bnb,
).eval()

# ---------- Build initial player/system context ----------
player = load_player(PLAYER_PATH)
system_text = " ".join([
    f"You are {player.get('name','Assistant')}, {player.get('profession','')}.",
    f"Your personality: {player.get('personality','')}",
    f"Your communication style: {player.get('style','')}",
    "Keep facts accurate." if player.get("facts_guard") else "",
    player.get("instructions",""),
])
asst_label = _first_word(player.get("name"))

app = FastAPI(title="Toddric Players API", version="1.1")


# ---------- Core generation ----------
def generate_response(message: str) -> str:
    """One-shot text generation."""
    msgs = [
        {"role": "system", "content": system_text},
        {"role": "user", "content": message},
    ]
    prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = tok([prompt], return_tensors="pt").to(mdl.device)
    with torch.inference_mode():
        out = mdl.generate(
            **inputs,
            max_new_tokens=120,
            temperature=0.3,
            top_p=0.9,
            do_sample=True,
            use_cache=True,
            eos_token_id=tok.eos_token_id,
            pad_token_id=tok.pad_token_id,
        )
    text = tok.decode(out[0, inputs["input_ids"].shape[-1]:], skip_special_tokens=True).strip()
    return text


async def stream_response(message: str) -> AsyncGenerator[str, None]:
    """Token streaming for chat UI."""
    msgs = [
        {"role": "system", "content": system_text},
        {"role": "user", "content": message},
    ]
    prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = tok([prompt], return_tensors="pt").to(mdl.device)

    streamer = TextIteratorStreamer(tok, skip_prompt=True, skip_special_tokens=True)
    thread = threading.Thread(
        target=lambda: mdl.generate(
            **inputs,
            streamer=streamer,
            max_new_tokens=120,
            temperature=0.3,
            top_p=0.9,
            do_sample=True,
            use_cache=True,
            eos_token_id=tok.eos_token_id,
            pad_token_id=tok.pad_token_id,
        ),
        daemon=True,
    )
    thread.start()
    for token in streamer:
        yield token


# ---------- API endpoints ----------
@app.get("/players")
async def list_players():
    """List available player JSONs."""
    base = os.path.dirname(PLAYER_PATH) or "."
    players = [f[:-5] for f in os.listdir(base) if f.endswith(".json")]
    return {"players": players}


@app.post("/player/{name}")
async def switch_player(name: str):
    """Switch active player/persona."""
    global player, system_text, asst_label
    new_p = load_player(name, base_dir=os.path.dirname(PLAYER_PATH) or ".")
    if not new_p:
        raise HTTPException(status_code=404, detail=f"Player '{name}' not found.")
    player = new_p
    system_text = " ".join([
        f"You are {new_p.get('name','Assistant')}, {new_p.get('profession','')}.",
        f"Your personality: {new_p.get('personality','')}",
        f"Your communication style: {new_p.get('style','')}",
        "Keep facts accurate." if new_p.get("facts_guard") else "",
        new_p.get("instructions",""),
    ])
    asst_label = _first_word(new_p.get("name"))
    print(f"🎭 Player switched to {player.get('name')}")
    return {"ok": True, "player": player.get("name")}


@app.post("/chat")
async def chat(payload: Dict[str, str]):
    """Synchronous chat endpoint."""
    message = payload.get("message", "").strip()
    if not message:
        raise HTTPException(status_code=400, detail="Empty message.")
    reply = generate_response(message)
    return {"player": player.get("name"), "reply": reply}


@app.post("/chat/stream")
async def chat_stream(payload: Dict[str, str]):
    """Streamed chat endpoint."""
    message = payload.get("message", "").strip()
    if not message:
        raise HTTPException(status_code=400, detail="Empty message.")
    return StreamingResponse(stream_response(message), media_type="text/plain")


# ---------- Serve static chat UI ----------
app.mount("/", StaticFiles(directory="static", html=True), name="static")

