# # backend/services/pdf_cluster.py
# from __future__ import annotations

# import os
# import tempfile
# import uuid
# from pathlib import Path
# import yaml
# from typing import Tuple, List

# from .s3_utils import list_s3_pdfs, download_to_path, upload_file
# from tableai_plugins import cluster_dir

# DATA_DIR = Path(os.getenv("DATA_DIR", "/data/pdfs")).resolve()

# # Lazy import to avoid loading the DiT model at app import time
# def _cluster_dir(pdf_dir: str, out_yaml: str, min_cluster: int):
#     from tableai_plugins import cluster_dir  # will init HF/torch on first call
#     return cluster_dir(pdf_dir, out_yaml, min_cluster=min_cluster)

# # --- settings ---
# # Keep simple; switch to Pydantic Settings if you prefer
# _DATA_DIR = Path(os.getenv("DATA_DIR", "/data/pdfs")).resolve()

# def ensure_data_dir() -> Path:
#     _DATA_DIR.mkdir(parents=True, exist_ok=True)
#     return _DATA_DIR

# def get_data_dir() -> Path:
#     ensure_data_dir()
#     return _DATA_DIR

# # --- runner ---
# def run_classification(data_dir: Path, *, min_cluster: int = 4) -> str:
#     """
#     Runs HDBSCAN clustering using the tableai_plugins.cluster_dir and returns the YAML path.
#     """
#     ensure_data_dir()
#     fd, yaml_path = tempfile.mkstemp(suffix=".yaml")
#     os.close(fd)
#     _cluster_dir(str(data_dir), yaml_path, min_cluster=min_cluster)
#     return yaml_path


# def _cluster_dir(pdf_dir: str, out_yaml: str, min_cluster: int):
#     # lazy import (loads DiT on first call)
#     from tableai_plugins import cluster_dir
#     return cluster_dir(pdf_dir, out_yaml, min_cluster=min_cluster)

# def run_classification_local_dir(pdf_dir: Path, *, min_cluster: int = 4) -> str:
#     pdf_dir = pdf_dir.resolve()
#     pdf_dir.mkdir(parents=True, exist_ok=True)
#     fd, yaml_path = tempfile.mkstemp(suffix=".yaml")
#     os.close(fd)
#     _cluster_dir(str(pdf_dir), yaml_path, min_cluster=min_cluster)
#     return yaml_path

# def run_classification_from_s3(
#     bucket: str,
#     prefix: str,
#     *,
#     min_cluster: int = 4,
#     keep_local: bool = False,
#     upload_yaml_to_s3: bool = True,
#     yaml_s3_key: str | None = None,
# ) -> Tuple[str, List[str]]:
#     """
#     Download PDFs from s3://bucket/prefix, classify, rewrite YAML to s3:// URIs,
#     and optionally upload clusters.yaml back to S3. Returns (yaml_path, s3_keys_used).
#     """
#     # 1) Download PDFs to temp staging dir
#     session_dir = DATA_DIR / f"s3_stage_{uuid.uuid4().hex}"
#     session_dir.mkdir(parents=True, exist_ok=True)

#     keys = list_s3_pdfs(bucket, prefix)
#     local_map = []  # (local_path, s3_uri)
#     for key in keys:
#         local_path = session_dir / Path(key).name
#         download_to_path(bucket, key, local_path)
#         local_map.append((local_path, f"s3://{bucket}/{key}"))

#     # 2) Classify locally
#     yaml_path = run_classification_local_dir(session_dir, min_cluster=min_cluster)

#     # 3) Rewrite file paths to S3 URIs
#     with open(yaml_path, "r") as f:
#         data = yaml.safe_load(f) or {}

#     s3_by_name = {Path(lp).name: s3uri for lp, s3uri in local_map}
#     for doc in data.get("documents", []):
#         fname = Path(doc["file"]).name
#         if fname in s3_by_name:
#             doc["file"] = s3_by_name[fname]

#     with open(yaml_path, "w") as f:
#         yaml.safe_dump(data, f, sort_keys=False)

#     # 4) Upload YAML back to S3 (optional)
#     if upload_yaml_to_s3:
#         key = yaml_s3_key or f"{prefix.rstrip('/')}/clusters.yaml"
#         upload_file(bucket, key, Path(yaml_path), content_type="text/yaml")

#     # 5) Clean up (optional)
#     if not keep_local:
#         try:
#             for p in session_dir.glob("*.pdf"):
#                 p.unlink(missing_ok=True)
#             session_dir.rmdir()
#         except Exception:
#             pass

#     return yaml_path, keys
