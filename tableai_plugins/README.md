# tableai (core)

Lightweight core library using pure Python + PyMuPDF heuristics to detect table-like regions.

## Install
```bash
pip install tableai                   # core only
pip install "tableai[all-clients]"    # HTTP clients
pip install "tableai[yolo-local]"     # local YOLO (install torch CPU/CUDA separately)
pip install "tableai[classifier-local]"
```

## Dev (uv)
```bash
uv sync --extra yolo-local --group yolo-cpu
# or CUDA 11.8:
uv sync --extra yolo-local --group yolo-cu118
```

## API
- `extract_basic(pdf_path) -> ExtractionResult`
- Optional local inference (lazy import):
  - `detect_tables_yolo_local(pdf_path, page=None, zoom=2.0, model_name="keremberke")`
  - `detect_tables_yolo_local_bytes(pdf_bytes, ...)`
  - `classify_pdf_local_bytes(pdf_bytes)` (stub to customize)
