#!/usr/bin/env bash
set -euo pipefail

# Activate the venv you use for serving
# adjust if your venv path differs
source "${HOME}/venvs/trainingEnv/bin/activate"

# Choose the FAST (bf16) merged model for SMS responsiveness
export TODDRIC_MODEL="${TODDRIC_MODEL:-/home/todd/training/ckpts/toddric-3b-merged-v3}"

# Perf knobs (safe defaults)
export TODDRIC_DEVICE_MAP='{"":0}'
export TODDRIC_ATTN="eager"
export TODDRIC_ALLOW_DOMAINS="youtube.com,youtu.be"
export TODDRIC_SMS_MAXNEW="${TODDRIC_SMS_MAXNEW:-60}"

# Run the API (single worker keeps model in one process)
exec uvicorn app_toddric:app --host 0.0.0.0 --port 8000 --workers 1 --log-level info --access-log
