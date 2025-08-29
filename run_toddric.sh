# on Rose, from the folder with app_toddric.py
source ../venvs/twilEnv/env/bin/activate
export TODDRIC_MODEL_DIR="/home/todd/training/ckpts/toddric-3b-merged-v3-bnb4"
export TODDRIC_KEY=Du25zavvslnemrmDHY1X5qwO505g0OO7DgTMpO41V6A=
# Optional tuning:
export TODDRIC_4BIT=0           # set 1 if you want 4-bit
export TODDRIC_DTYPE=bf16       # or fp16 if needed
export TODDRIC_MAX_NEW=200
export REPLY_MAX=480
uvicorn app_toddric:app --host 0.0.0.0 --port 8000

