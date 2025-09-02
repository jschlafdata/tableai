# tableai — UV Starter (Optional Extras + Plugins)

This repo demonstrates a **lightweight core** library (`tableai-slim`) plus **optional** local inference and microservices,
wired with **uv workspaces**. Pip users can install only what they need via **extras**.

## Layout
- `tableai` – core package (PyMuPDF-based table hints). Optional extras for **local** YOLO/Classifier and **HTTP clients**.
- `services/backend` – FastAPI backend that orchestrates core + optional plugins (HTTP) or local inference fallback.
- `plugins/yolo` – YOLO microservice (CPU & GPU Dockerfiles, uv-based).
- `plugins/classifier` – Classifier microservice (CPU & GPU Dockerfiles, uv-based).
- `frontend/` – optional React app (points to backend).

## Quickstart
```bash
uv sync
# Dev: run backend
uv run --package backend uvicorn backend.main:app --host 0.0.0.0 --port 8011
```

<!-- cache-keys = [{ file = "pyproject.toml" }, { file = "requirements.txt" }] -->
<!-- cache-keys = [{ file = "**/*.toml" }] -->
<!--  -->
<!-- cache-keys = [{ file = "pyproject.toml" }, { env = "MY_ENV_VAR" }] -->
<!-- reinstall-package = ["my-package"] -->
<!--  -->



### Docker Compose
CPU-only:
```bash
docker compose up --build
# backend    → http://localhost:8011
# yolo cpu   → http://localhost:8001/health
# classifier → http://localhost:8005/health
```
GPU (requires NVIDIA Container Toolkit):
```bash
docker compose --profile gpu up --build
```

## Optional extras (pip)
```bash
pip install tableai                     # core only
pip install "tableai[all-clients]"      # HTTP clients (tiny)
pip install "tableai[yolo-local]"       # local YOLO; install torch CPU/CUDA separately
pip install "tableai[classifier-local]"
```

### uv convenience for devs
```bash
uv sync --extra yolo-local --group yolo-cpu
# or:
uv sync --extra yolo-local --group yolo-cu118
```

## Notes
- **Core** never pulls heavy ML libs by default.
- **Backend** falls back to **local** inference only if extras are present and HTTP plugins are not healthy.
- **Services** are container-first and independent; not required by pip installs.
