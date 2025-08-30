# --- Base: Python slim (works with NVIDIA Container Toolkit on host) ---
FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ca-certificates build-essential git \
 && rm -rf /var/lib/apt/lists/*

# --- Python deps layer ---
# We keep torch install flexible (GPU vs CPU) via build args
# Provide your own wheel/index at build time if you want CUDA Torch.
ARG TORCH_SPEC=""        # e.g. 'torch torchvision --index-url https://download.pytorch.org/whl/cu124'
ARG EXTRA_PIP=""

# Minimal requirements (no torch here by default)
# (bitsandbytes optional; install only if you plan 4-bit)
RUN python -m pip install --upgrade pip && \
    pip install \
      uvicorn[standard] fastapi httpx \
      transformers accelerate \
      "bitsandbytes>=0.43.0" \
      ${EXTRA_PIP} \
    && if [ -n "${TORCH_SPEC}" ]; then \
         echo "Installing Torch via: ${TORCH_SPEC}" && pip install ${TORCH_SPEC}; \
       else \
         echo "No TORCH_SPEC provided — you can still run CPU or mount host GPU libs"; \
       fi

# App code in /app
COPY app_toddric.py /app/app_toddric.py

# Healthcheck baked in
HEALTHCHECK --interval=30s --timeout=5s --retries=5 CMD curl -fsS http://127.0.0.1:8000/healthz || exit 1

EXPOSE 8000

# Default runtime env; override via compose/.env
ENV TODDRIC_MODEL_DIR=/models \
    TODDRIC_DTYPE=bf16 \
    TODDRIC_MAX_NEW=200 \
    REPLY_MAX=480 \
    TODDRIC_4BIT=0 \
    UVICORN_WORKERS=1

CMD ["uvicorn", "app_toddric:app", "--host", "0.0.0.0", "--port", "8000"]

