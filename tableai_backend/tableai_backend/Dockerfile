FROM python:3.12-slim-bullseye

ENV PIP_NO_CACHE_DIR=off \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    UV_SYSTEM_PYTHON=1 \
    MODEL_CACHE_DIR=/models \
    PORT=8001 \
    TABLEAI_BUILD=0

COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

ENV PATH="/app/.venv/bin:$PATH"
ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy

# Base OS deps and CA setup
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    git \
    curl && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . .
RUN uv pip install -e .

# Expose port and define the command to run your application
EXPOSE 8000
CMD ["uvicorn", "tableai_backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]