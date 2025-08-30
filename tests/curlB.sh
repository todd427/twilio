curl -s -X POST http://127.0.0.1:8000/chat \
  -H 'content-type: application/json' -H 'Authorization: Bearer 7ff03c53-5529-43a2-b549-695957c8d161' \
  -d '{"message":"Say hi in one sentence.","session_id":"test","style":"sms_short","max_new_tokens":48}'
# → JSON reply in < 2–4s
