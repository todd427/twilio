8# main.py
# Twilio Voice → FastAPI → your LLM on :8000
# Run:  uvicorn main:app --host 0.0.0.0 --port 8100
# Tunnel: cloudflared tunnel --url http://localhost:8100

import os, re
from fastapi import FastAPI, Request, Response
from twilio.twiml.voice_response import VoiceResponse, Gather, Pause
import httpx

app = FastAPI()

# ---- Config (override via env) ---------------------------------------------
GEN_URL        = os.getenv("GEN_URL", "http://localhost:8000/api/generate")
VOICE          = os.getenv("VOICE", "Polly.Amy-Neural")  # warm British female
LANG           = os.getenv("LANG", "en-GB")
MAX_TTS_CHARS  = int(os.getenv("MAX_TTS_CHARS", "1500"))
GATHER_TIMEOUT = int(os.getenv("GATHER_TIMEOUT", "7"))   # secs of no input
SPEECH_TIMEOUT = os.getenv("SPEECH_TIMEOUT", "6")       # secs after speech stops
PROMPT_PAUSE   = int(os.getenv("PROMPT_PAUSE", "1"))    # secs before prompts
# ---------------------------------------------------------------------------

def _twiml(vr: VoiceResponse) -> Response:
    return Response(content=str(vr), media_type="application/xml")

def _clean_text(s: str) -> str:
    if not s:
        return "Okay."
    s = re.sub(r"<[^>]+>", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s[:MAX_TTS_CHARS]

async def ask_toddric(user_text: str) -> str:
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(20.0)) as client:
            r = await client.post(
                GEN_URL,
                json={"text": user_text, "user_id": "voice-caller"},
                headers={"Accept": "application/json"},
            )
            r.raise_for_status()
            data = r.json()
            return _clean_text(data.get("reply") or "Okay.")
    except Exception as e:
        return _clean_text(f"Sorry, I hit an error talking to the model: {e}")

def _gather_block(prompt: str) -> Gather:
    g = Gather(
        input="speech",
        action="/handleCall",
        method="POST",
        timeout=GATHER_TIMEOUT,
        speech_timeout=SPEECH_TIMEOUT,  # <- more patient
        language=LANG,
    )
    # brief pause so we don't talk over the caller as they breathe/think
    g.append(Pause(length=PROMPT_PAUSE))
    g.say(prompt, voice=VOICE)
    return g

@app.get("/health")
async def health():
    return {
        "ok": True,
        "gen_url": GEN_URL,
        "voice": VOICE,
        "lang": LANG,
        "gather_timeout": GATHER_TIMEOUT,
        "speech_timeout": SPEECH_TIMEOUT,
    }

# Handy GET to preview TwiML in a browser
@app.get("/handleCall")
async def handle_call_get():
    vr = VoiceResponse()
    vr.append(_gather_block("Hi, this is Toddric. Ask me anything, then pause."))
    # Softer fallback if truly no input
    vr.say("No rush. When you're ready, just speak.", voice=VOICE)
    return _twiml(vr)

@app.post("/handleCall")
async def handle_call(request: Request):
    form = await request.form()
    speech = form.get("SpeechResult")

    vr = VoiceResponse()

    if not speech:
        # First turn or silence: be gentle and patient
        vr.append(_gather_block("Hi, this is Toddric. Ask me anything, then pause."))
        vr.say("Take your time.", voice=VOICE)
        return _twiml(vr)

    # We have a transcript -> reply, then re-gather
    reply = await ask_toddric(speech)
    vr.say(reply, voice=VOICE)
    vr.append(_gather_block("Go ahead."))
    return _twiml(vr)

