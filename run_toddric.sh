#!/usr/bin/env bash
set -euo pipefail

# Activate the venv you use for serving
# adjust if your venv path differs
source "${HOME}/venvs/trainingEnv/bin/activate"

# Choose the FAST (bf16) merged model for SMS responsiveness
export TODDRIC_MODEL="${TODDRIC_MODEL:-/home/todd/training/ckpts/toddric-1_5b-merged-v1}"
export RL_MAX=30
export RL_WINDOW=300
# Perf knobs (safe defaults)
export TODDRIC_DEVICE_MAP='{"":0}'
export TODDRIC_ATTN="eager"
export TODDRIC_ALLOW_DOMAINS="youtube.com,youtu.be"
export TODDRIC_SMS_MAXNEW="${TODDRIC_SMS_MAXNEW:-60}"
#
export TODDRIC_BEARER=7ff03c53-5529-43a2-b549-695957c8d161
export TOKEN_COOKIE_SECURE=1



# Run the API (single worker keeps model in one process)
#exec uvicorn app_toddric:app --host 0.0.0.0 --port 8000 --workers 1 --log-level info --access-log
exec uvicorn app_toddric:app \
  --host 0.0.0.0 \
  --port 8000 \
  --workers 1 \
  --timeout-keep-alive 30 \
  --loop uvloop \
  --http h11 \
  --log-level info \
  --access-log

