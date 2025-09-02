FROM python:3.11-slim

FROM python:3.12-slim-bullseye

ENV PIP_NO_CACHE_DIR=off \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    PORT=8001 \
    UV_SYSTEM_PYTHON=1 \
    MODEL_CACHE_DIR=/models \
    TABLEAI_BUILD=0

COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv
WORKDIR /app

ENV PATH="/app/.venv/bin:$PATH"
ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy

# Base OS deps and CA setup
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential && \
    libpq-dev && \
    rm -rf /var/lib/apt/lists/*


WORKDIR /app
COPY . .

RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --frozen --no-install-project

COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt
COPY . /app

RUN uv add uvicorn==0.35.0 --extra standard
# RUN uv sync --extra cpu

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync

RUN uv pip list
EXPOSE $PORT

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:$PORT/health || exit 1

CMD ["uv", "run", "uvicorn", "plugins.yolo.src.app.main:app", "--host", "0.0.0.0", "--port", "8001"]