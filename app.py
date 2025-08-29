import os
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
import httpx

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
SYSTEM_PROMPT = os.getenv("SYSTEM_PROMPT", "You are Toddric. Be concise, helpful, kind, and SMS-friendly.")
REPLY_MAX = int(os.getenv("REPLY_MAX", "480"))
REQUIRE_BEARER = os.getenv("TODDRIC_KEY", "")  # optional shared secret

app = FastAPI(title="Toddric SMS API")

class InMsg(BaseModel):
    user_id: str
    message: str
    channel: str = "sms"

class OutMsg(BaseModel):
    reply: str

@app.post("/v1/reply", response_model=OutMsg)
async def reply(inmsg: InMsg, authorization: str | None = Header(default=None)):
    # Optional shared-secret check
    if REQUIRE_BEARER:
        if not authorization or not authorization.startswith("Bearer ") or authorization.split(" ", 1)[1] != REQUIRE_BEARER:
            raise HTTPException(status_code=401, detail="Unauthorized")

    user_text = (inmsg.message or "").strip()
    if not OPENAI_API_KEY:
        # echo mode if no key set (still useful for plumbing)
        return {"reply": f"Echo: {user_text}"[:REPLY_MAX]}

    try:
        async with httpx.AsyncClient(timeout=12) as client:
            r = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {OPENAI_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": MODEL,
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_text},
                    ],
                    "temperature": 0.7,
                    "max_tokens": 200,
                },
            )
        if r.status_code != 200:
            # bubble up reason so you can see it during setup
            raise HTTPException(status_code=502, detail=f"openai:{r.status_code}:{r.text[:300]}")
        data = r.json()
        txt = (data.get("choices", [{}])[0].get("message", {}).get("content") or "...").strip()
        return {"reply": txt[:REPLY_MAX]}
    except httpx.RequestError as e:
        raise HTTPException(status_code=504, detail=f"timeout:{e}")  # network/timeout
