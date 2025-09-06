# tableai_plugins/dit.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, List, Optional, Tuple, Union
from pathlib import Path
import io
import os

import numpy as np
import yaml

from .loader import MissingOptionalDependency

ImageLike = Union["PIL.Image.Image", np.ndarray, str, Path]

@dataclass
class DiTPlugin:
    """
    Dynamic DiT plugin for image/PDF embeddings + optional clustering.

    Usage:
        from tableai_plugins import get_plugin
        DiT = get_plugin("dit")
        dit = DiT()  # loads model on init
        vec = dit.embed("doc.pdf")  # CLS embedding for first page
        docs, emb2d, labels = dit.cluster_pdfs("/path/to/dir", out_yaml="clusters.yaml")
    """
    name: str = "dit"
    model_id: str = "microsoft/dit-base-finetuned-rvlcdip"
    device: Optional[str] = None  # "cuda" / "cpu"

    # Internal handles after init
    _torch: Any = None
    _processor: Any = None
    _model: Any = None
    _device: str = "cpu"

    def __post_init__(self):
        # Heavy deps load here
        try:
            import torch as _torch  # type: ignore
        except Exception as e:
            raise MissingOptionalDependency(
                "PyTorch is required for the DiT plugin."
            ) from e
        try:
            from transformers import AutoImageProcessor, AutoModel  # type: ignore
        except Exception as e:
            raise MissingOptionalDependency(
                "transformers is required for the DiT plugin."
            ) from e

        self._torch = _torch
        self._device = self.device or ("cuda" if _torch.cuda.is_available() else "cpu")
        self._processor = AutoImageProcessor.from_pretrained(self.model_id)
        self._model = AutoModel.from_pretrained(self.model_id).to(self._device).eval()

    # ------------------------------- Public API -------------------------------

    def embed(self, image_or_path: ImageLike) -> np.ndarray:
        """
        Return a 1D CLS embedding (np.ndarray) for a PIL Image, np.ndarray,
        or a path to an image/PDF (first page).
        """
        pil = self._to_pil(image_or_path)
        inputs = self._processor(images=pil, return_tensors="pt").to(self._device)
        with self._torch.inference_mode():
            out = self._model(**inputs).last_hidden_state  # [1, seq_len, hidden]
        return out[:, 0, :].detach().cpu().numpy().squeeze()  # CLS token

    def batch_embed(self, images: Iterable[ImageLike]) -> np.ndarray:
        vecs: List[np.ndarray] = [self.embed(x) for x in images]
        if not vecs:
            return np.empty((0, 0), dtype=np.float32)
        return np.vstack(vecs)

    def cluster_pdfs(
        self,
        pdf_dir: Union[str, Path],
        out_yaml: Optional[Union[str, Path]] = None,
        min_cluster_size: int = 4,
    ) -> Tuple[List[dict], np.ndarray, np.ndarray]:
        """
        Render 1st page of PDFs in a directory, embed with DiT, reduce with UMAP, cluster with HDBSCAN.
        Returns: (docs, emb2d, labels)
            docs[i] = {"file": <path>, "cluster": int, "umap_x": float, "umap_y": float}
        If out_yaml is not None, write {"documents": docs} to that file.
        """
        try:
            import umap  # type: ignore
            import hdbscan  # type: ignore
        except Exception as e:
            raise MissingOptionalDependency(
                "umap-learn and hdbscan are required for clustering.\n"
                "Install with: pip install 'tableai-plugins[dit]'"
            ) from e
        try:
            from tqdm import tqdm  # type: ignore
        except Exception:
            # Soft fallback if tqdm isn't present (but you include it in extras)
            def tqdm(x, **kwargs): return x

        pdf_dir = Path(pdf_dir)
        files = sorted([p for p in pdf_dir.iterdir() if p.suffix.lower() == ".pdf"])
        paths, feats = [], []
        for p in tqdm(files, desc="PDFs", unit="pdf"):
            pil = self._pdf_first_page_to_pil(p)
            vec = self.embed(pil)
            paths.append(str(p))
            feats.append(vec)

        if not feats:
            return [], np.empty((0, 2), dtype=np.float32), np.empty((0,), dtype=np.int32)

        X = np.vstack(feats)
        reducer = umap.UMAP(n_neighbors=15, min_dist=0.05, random_state=42)
        emb2d = reducer.fit_transform(X)

        clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, metric="euclidean")
        labels = clusterer.fit_predict(X)  # -1 = noise

        docs = [
            dict(file=path, cluster=int(lbl), umap_x=float(x), umap_y=float(y))
            for path, lbl, (x, y) in zip(paths, labels, emb2d)
        ]

        if out_yaml:
            with open(out_yaml, "w", encoding="utf-8") as f:
                yaml.safe_dump({"documents": docs}, f, sort_keys=False)

        return docs, emb2d, labels

    # ------------------------------- Internals --------------------------------

    def _to_pil(self, x: ImageLike) -> "PIL.Image.Image":
        try:
            from PIL import Image  # type: ignore
        except Exception as e:
            raise MissingOptionalDependency(
                "Pillow is required to handle images for the DiT plugin."
            ) from e

        if isinstance(x, Image.Image):
            return x
        if isinstance(x, np.ndarray):
            return Image.fromarray(x)
        if isinstance(x, (str, Path)):
            p = Path(x)
            if p.suffix.lower() == ".pdf":
                return self._pdf_first_page_to_pil(p)
            # Assume image file
            return Image.open(p).convert("RGB")
        raise TypeError(f"Unsupported image type: {type(x)!r}")

    @staticmethod
    def _pdf_first_page_to_pil(pdf_path: Union[str, Path], zoom: float = 2.0) -> "PIL.Image.Image":
        try:
            import fitz  # PyMuPDF  # type: ignore
        except Exception as e:
            raise MissingOptionalDependency(
                "PyMuPDF (fitz) is required to render PDF pages for the DiT plugin.\n"
                "Install with: pip install pymupdf"
            ) from e
        from PIL import Image  # safe import here

        doc = fitz.open(str(pdf_path))
        page = doc.load_page(0)
        pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom))
        return Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
