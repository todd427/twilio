# VOICES.md

Set up **Toddric Voice** (Twilio → FastAPI → your LLM) alongside your existing **SMS/WhatsApp** bot.

---

## Overview

- **Voice bot** (this repo): FastAPI app served on **:8100** (`app:main`), exposed via **Cloudflare Tunnel**, answering phone calls with TTS + speech recognition.
- **SMS/WhatsApp bot** (existing): FastAPI app served on **:8000**, same machine, already wired to Twilio Messaging.

High-level flow:

```
Caller → Twilio Voice → [https://<your-tunnel>/handleCall]
                      → FastAPI :8100 → (POST) http://localhost:8000/api/generate
                      → LLM reply → Twilio speaks with Polly.Amy-Neural
```

---

## Prerequisites

- Python 3.10+ with:
  ```bash
  pip install fastapi uvicorn twilio httpx
  ```
- Your **LLM API** (or SMS handler) already running on **:8000**
  - Recommended: expose `POST /api/generate` that accepts `{"text": "...", "user_id": "..."}` and returns `{"reply": "..."}`.
- A **Twilio** account with a phone number capable of **Voice**.
- **cloudflared** installed (or ngrok).

---

## Ports & Processes

- **:8000** — existing SMS/WhatsApp app (Messaging webhook).
- **:8100** — new Voice app (Voice webhook).

> Keep them **separate** so voice experiments don’t collide with your messaging bot.

---

## Run the Voice App (on :8100)

Your `main.py` is already prepared with a warm British female voice:

```bash
export GEN_URL="http://localhost:8000/api/generate"
export VOICE="Polly.Amy-Neural"   # change if desired
export LANG="en-GB"
export GATHER_TIMEOUT=7           # patience tuning (seconds)
export SPEECH_TIMEOUT=6

uvicorn main:app --host 0.0.0.0 --port 8100
```

Health check:
```bash
curl -s http://127.0.0.1:8100/health
```

---

## Expose :8100 with a Tunnel

### Quick (ephemeral) tunnel
```bash
cloudflared tunnel --url http://localhost:8100
```
Copy the printed URL, e.g.:
```
https://club-abc-voice.trycloudflare.com
```

### Stable (named) tunnel (recommended)
```bash
cloudflared tunnel login
cloudflared tunnel create toddric-voice
cloudflared tunnel route dns toddric-voice voice.yourdomain.com
cloudflared tunnel run toddric-voice --url http://localhost:8100
```
Use `https://voice.yourdomain.com` in Twilio so your URL **doesn’t change**.

---

## Wire Twilio (Voice)

1. In Twilio Console → **Phone Numbers → Active numbers → (select your number)**  
2. **Voice & Fax** section → **A CALL COMES IN**  
   - **Webhook**  
   - **URL**: `https://<your-voice-url>/handleCall`  
   - **Method**: `HTTP POST`  
3. Save.

Validate from your machine:
```bash
curl -i https://<your-voice-url>/health
curl -i https://<your-voice-url>/handleCall       # GET shows TwiML (dev helper)
curl -i -X POST https://<your-voice-url>/handleCall -d '' \
  -H 'Content-Type: application/x-www-form-urlencoded'
```

---

## Keep SMS/WhatsApp on :8000

- **Messaging webhook** (already in place):
  - Twilio Console → your number (or WhatsApp sender) → **Messaging**
  - **A MESSAGE COMES IN** → `https://<your-existing-url>/handleMessage` (POST)
- Your voice app calls `GEN_URL=http://localhost:8000/api/generate` internally so both **share the same brain**.

> No change required for your existing :8000 bot unless you still need to add `/api/generate`. See Appendix below for a tiny example route.

---

## Changing the Voice

In `main.py` we default to **Polly.Amy-Neural** (warm British female).

Options:
- `VOICE="Polly.Amy-Neural"` (current)
- `VOICE="Polly.Emma-Neural"` (slightly softer en-GB female)
- You can also use Google voices, e.g. `VOICE="Google.en-GB-Wavenet-B"`

Update and restart:
```bash
export VOICE="Polly.Emma-Neural"
uvicorn main:app --host 0.0.0.0 --port 8100
```

> Keep `LANG="en-GB"` so speech recognition matches the accent.

---

## Patience / Turn-taking

If the bot feels “impatient,” tune these env vars:

```bash
export GATHER_TIMEOUT=9   # secs of silence before Twilio ends the listen
export SPEECH_TIMEOUT=7   # secs Twilio waits after caller stops talking
```

These map to `<Gather timeout>` and `<Gather speech_timeout>`.

---

## Troubleshooting

**Symptom** | **Likely Cause** | **Fix**
---|---|---
`Error 11200` with HTTP 502 | Tunnel URL changed / tunnel down | Update Twilio with the **current** tunnel URL; restart `cloudflared`; try named tunnel.
Call answers then hangs | `/handleCall` 405/404 | Ensure **POST** route exists at `/handleCall`; our app provides both GET and POST for dev convenience.
Long delay / Twilio timeout | LLM too slow in webhook path | First turn avoids LLM; for later turns keep model fast; consider caching, shorter max tokens.
No sound / wrong accent | Voice name unsupported / lang mismatch | Try `Polly.Emma-Neural`; keep `LANG=en-GB`.
Works locally, not via tunnel | Firewall/NAT or cloudflared hiccup | Test with `curl` against the public URL; try ngrok to isolate.

Quick sanity:
```bash
# App listening?
ss -ltnp | grep 8100
# Local health
curl -i http://127.0.0.1:8100/health
# Public health
curl -i https://<your-voice-url>/health
```

---

## Operational Tips

- **Separate ports**: :8000 (Messaging) vs :8100 (Voice).
- **Respond fast**: Twilio expects XML within ~15s; keep heavy LLM work short or pre-warm model.
- **Logs**: run both with `--log-level info` and watch `cloudflared --loglevel info`.
- **Security** (later): Validate Twilio signatures on inbound webhooks before trusting requests.

---

## Appendix A — Minimal `/api/generate` for :8000

If your :8000 app doesn’t have a JSON generation endpoint yet, add this to it:

```python
# Inside your :8000 app
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class GenIn(BaseModel):
    text: str
    user_id: str | None = None

@app.post("/api/generate")
async def generate(inp: GenIn):
    # TODO: call your real LLM here; this is a stub
    reply = f"You said: {inp.text}. I'm a demo running on :8000."
    return {"reply": reply}
```

Now the voice app on :8100 can call:
```
GEN_URL=http://localhost:8000/api/generate
```

---

## Appendix B — Quick Voice Test Script

To simulate Twilio’s POST:

```bash
VOICE_URL="https://<your-voice-url>/handleCall"

# First hit (no SpeechResult) should return a <Gather>
curl -i -X POST "$VOICE_URL" -H 'Content-Type: application/x-www-form-urlencoded' -d ''

# Simulate a turn with a transcript
curl -i -X POST "$VOICE_URL" -H 'Content-Type: application/x-www-form-urlencoded' \
  --data-urlencode 'SpeechResult=Tell me a story about Donegal.'
```

You should see `200 OK` and `Content-Type: application/xml` with `<Say>` in the body.

---

## Appendix C — Switching to Another Voice

Update the env var and restart:

```bash
export VOICE="Polly.Emma-Neural"   # or Google.en-GB-Wavenet-B
uvicorn main:app --host 0.0.0.0 --port 8100
```

No code change needed.
