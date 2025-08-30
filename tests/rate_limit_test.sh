for i in $(seq 1 35); do
  curl -s -o /dev/null -w "%{http_code}\n" -X POST http://127.0.0.1:8000/chat \
   -H 'content-type: application/json' -H 'Authorization: Bearer 7ff03c53-5529-43a2-b549-695957c8d161' \
   -d '{"message":"ping","session_id":"test","style":"sms_short","max_new_tokens":16}'
done
