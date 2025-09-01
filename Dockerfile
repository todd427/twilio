FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ca-certificates build-essential git \
 && rm -rf /var/lib/apt/lists/*

# Install base deps (no torch here)
COPY requirements.txt .
RUN python -m pip install --upgrade pip && pip install -r requirements.txt

# Optional Torch install controlled by build-arg (CUDA or CPU wheel)
ARG TORCH_SPEC=""
RUN if [ -n "${TORCH_SPEC}" ]; then \
      echo "Installing Torch via: ${TORCH_SPEC}" && pip install ${TORCH_SPEC}; \
    else \
      echo "Skipping Torch install here (you can pip install at runtime)"; \
    fi

# Copy your app code (so toddric_chat.py, etc. are included)
COPY . .

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --retries=5 CMD curl -fsS http://127.0.0.1:8000/healthz || exit 1

CMD ["uvicorn", "app_toddric:app", "--host", "0.0.0.0", "--port", "8000"]

