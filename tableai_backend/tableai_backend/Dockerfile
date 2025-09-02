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


RUN uv venv /opt/venv
COPY uv.lock pyproject.toml ./
RUN . /opt/venv/bin/activate && \
    uv sync --frozen

# --- Stage 2: Final Image ---

# Set up non-root user for security
RUN useradd --create-home --shell /bin/bash appuser
USER appuser
WORKDIR /home/appuser/app

# Copy your application's pre-built wheel and source code
# The wheel contains the version and metadata, so no git is needed.
COPY --chown=appuser:appuser dist/*.whl .
COPY --chown=appuser:appuser tableai_backend ./tableai_backend

# Set path to use the venv python and install the wheel
ENV PATH="/opt/venv/bin:$PATH"
RUN uv pip install *.whl

# Expose port and define the command to run your application
EXPOSE 8000
CMD ["uvicorn", "tableai_backend.main:app", "--host", "0.0.0.0", "--port", "8000"]