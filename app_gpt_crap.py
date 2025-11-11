import json
import os
from pathlib import Path
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
import torch
import threading
from fastapi.staticfiles import StaticFiles

# ==========================================================
# CONFIGURATION
# ==========================================================
MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct")
PLAYERS_DIR = os.getenv("PLAYERS_DIR", "players")
DEFAULT_PLAYER = os.getenv("DEFAULT_PLAYER", "kermit.json")

device = "cuda" if torch.cuda.is_available() else "cpu"

# ==========================================================
# APP SETUP
# ==========================================================
app = FastAPI(title="Toddric Players API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================================
# MODEL LOAD
# ==========================================================
print(f"🧩 Loading model {MODEL_NAME} …")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    trust_remote_code=True,
)
model.eval()
print(f"✅ Model loaded on {device}")

# ==========================================================
# PLAYER HANDLING
# ==========================================================
def load_player(name_or_path: str, base_dir: str = PLAYERS_DIR):
    """Load a player JSON by name or full path."""
    path = Path(name_or_path)
    if not path.exists():
        path = Path(base_dir) / (name_or_path if name_or_path.endswith(".json") else f"{name_or_path}.json")
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)

# Global state
player = load_player(DEFAULT_PLAYER, PLAYERS_DIR)
if not player:
    raise RuntimeError(f"Could not find default player: {DEFAULT_PLAYER}")
print(f"🎭 Default player: {player.get('name')}")

def build_system_text(p):
    """Build system context string for the player."""
    return (
        f"You are {p.get('name','Assistant')}, {p.get('profession','')}. "
        f"Your personality: {p.get('personality','')}. "
        f"Your communication style: {p.get('style','')}. "
        f"{'Keep facts accurate.' if p.get('facts_guard') else ''} "
        f"{p.get('instructions','')}"
    )

def _first_word(name: str) -> str:
    return name.split(" ")[0] if name else "Assistant"

# ==========================================================
# CHAT LOGIC
# ==========================================================
def generate_response(message: str):
    """Blocking generation."""
    global player
    system_text = build_system_text(player)
    print(f"🧠 Using player: {player.get('name')}")

    # Clean, role-separated prompt
    prompt = (
        f"System: {system_text}\n"
        f"User: {message}\n"
        f"{_first_word(player.get('name'))}:"
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=150,
            do_sample=True,
            top_p=0.9,
            temperature=0.6,
        )
    text = tokenizer.decode(output[0], skip_special_tokens=True)
    reply = text.split(f"{_first_word(player.get('name'))}:")[-1].strip()
    return reply


def stream_response(message: str):
    """Streaming response."""
    global player
    system_text = build_system_text(player)
    print(f"🧠 Using player: {player.get('name')}")

    prompt = (
        f"System: {system_text}\n"
        f"User: {message}\n"
        f"{_first_word(player.get('name'))}:"
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    streamer = TextIteratorStreamer(tokenizer, skip_special_tokens=True)
    gen_kwargs = dict(
        **inputs,
        streamer=streamer,
        max_new_tokens=150,
        do_sample=True,
        top_p=0.9,
        temperature=0.6,
    )

    thread = threading.Thread(target=model.generate, kwargs=gen_kwargs)
    thread.start()

    for new_text in streamer:
        yield new_text

# ==========================================================
# ROUTES
# ==========================================================
@app.get("/players")
def list_players():
    base = Path(PLAYERS_DIR)
    players = [p.stem for p in base.glob("*.json")]
    return {"players": players}


@app.get("/player")
def get_current_player():
    global player
    return {"current": player.get("name")}


@app.post("/player/{name}")
async def switch_player(name: str):
    global player
    new_p = load_player(name, PLAYERS_DIR)
    if not new_p:
        raise HTTPException(status_code=404, detail=f"Player '{name}' not found.")
    player = new_p
    print(f"🎭 Player switched to {player.get('name')}")
    return {"ok": True, "player": player.get("name")}


@app.post("/chat")
async def chat(request: Request):
    data = await request.json()
    message = data.get("message", "")
    if not message:
        raise HTTPException(status_code=400, detail="Missing 'message'")
    reply = generate_response(message)
    return {"reply": reply}


@app.post("/chat/stream")
async def chat_stream(request: Request):
    data = await request.json()
    message = data.get("message", "")
    if not message:
        raise HTTPException(status_code=400, detail="Missing 'message'")
    return StreamingResponse(stream_response(message), media_type="text/plain")

# ==========================================================
# STATIC FRONTEND
# ==========================================================
app.mount("/", StaticFiles(directory="static", html=True), name="static")

# ==========================================================
# MAIN
# ==========================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app_players:app", host="0.0.0.0", port=8020, reload=True)

