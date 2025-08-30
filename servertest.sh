curl -m 9 -sS -H 'content-type: application/json' \
  -d '{"message":"What do you know about Jimi Hendrix?","session_id":"test","max_new_tokens":80,"temperature":0.2,"top_p":0.9,"style":"sms_short","instruction":"Reply in 2–3 concise sentences, SMS-friendly."}' \
  https://club-lcd-invest-pound.trycloudflare.com/chat

