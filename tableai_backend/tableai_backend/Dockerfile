FROM python:3.12-slim-bullseye

ENV PIP_NO_CACHE_DIR=off \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    UV_SYSTEM_PYTHON=1 \
    MODEL_CACHE_DIR=/models \
    PORT=8001 \
    TABLEAI_BUILD=0

WORKDIR /app
COPY .git .git

COPY --from=ghcr.io/astral-sh/uv:0.8.13 /uv /uvx /bin/

# Base OS deps and CA setup
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    git \
    curl && \
    rm -rf /var/lib/apt/lists/*


# Place executables in the environment at the front of the path
# Ref: https://docs.astral.sh/uv/guides/integration/docker/#using-the-environment
ENV PATH="/app/.venv/bin:$PATH"

# Compile bytecode
# Ref: https://docs.astral.sh/uv/guides/integration/docker/#compiling-bytecode
ENV UV_COMPILE_BYTECODE=1

# uv Cache
# Ref: https://docs.astral.sh/uv/guides/integration/docker/#caching
ENV UV_LINK_MODE=copy

COPY . .

# Install dependencies
# Ref: https://docs.astral.sh/uv/guides/integration/docker/#intermediate-layers
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --frozen --no-install-project

# --no-install-project

ENV PYTHONPATH=/app

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --all-extras

# Expose port and define the command to run your application
EXPOSE 8000
CMD ["uv", "run", "tableai-api"]

# CMD ["uv", "run", "uvicorn", "tableai_backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]