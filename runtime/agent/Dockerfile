FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_CACHE_DIR=/tmp/uv-cache \
    NEWS2ETF_REPO_ROOT=/app \
    NEWS2ETF_RUNTIME_ROOT=/app/runtime/agent \
    NEWS2ETF_SHARED_DATA_ROOT=/app/data

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml uv.lock /app/
RUN uv export \
    --frozen \
    --format requirements.txt \
    --only-group runtime \
    --no-emit-project \
    --no-editable \
    --output-file /tmp/runtime-requirements.txt \
    && python -m venv /opt/venv \
    && /opt/venv/bin/python -m pip install --upgrade pip \
    && /opt/venv/bin/pip install -r /tmp/runtime-requirements.txt

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PATH=/opt/venv/bin:$PATH \
    NEWS2ETF_REPO_ROOT=/app \
    NEWS2ETF_RUNTIME_ROOT=/app/runtime/agent \
    NEWS2ETF_SHARED_DATA_ROOT=/app/data

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /tmp/runtime-requirements.txt /app/runtime-requirements.txt
COPY runtime/agent /app/runtime/agent

RUN mkdir -p \
    /app/data \
    /app/runtime/agent/checkpoints \
    /app/runtime/agent/data \
    /app/runtime/agent/data/inputs \
    /app/runtime/agent/models \
    /app/runtime/agent/wandb \
    /app/runtime/agent/scripts

WORKDIR /app/runtime/agent

# Example:
#   docker build -f runtime/agent/Dockerfile -t news2etf-agent .
#   docker run --rm \
#     -e OPENAI_API_KEY=... \
#     -e WANDB_API_KEY=... \
#     -v $(pwd)/data:/app/data \
#     -v $(pwd)/runtime/agent/models:/app/runtime/agent/models \
#     -v $(pwd)/runtime/agent/data/inputs:/app/runtime/agent/data/inputs \
#     -v $(pwd)/runtime/agent/checkpoints:/app/runtime/agent/checkpoints \
#     news2etf-agent \
#     python main.py backtest --start-date 2024-01-01 --end-date 2024-12-31

CMD ["python", "main.py", "--help"]
