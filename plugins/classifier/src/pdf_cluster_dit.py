import os, io, yaml, argparse, fitz, torch
from PIL import Image
from tqdm import tqdm
import numpy as np
import hdbscan, umap
from transformers import AutoImageProcessor, AutoModel
from functools import partial
import sys

# ----------------------------------------------------------------------
# 1.  Set-up – load DiT
# ----------------------------------------------------------------------
MODEL_ID = "microsoft/dit-base-finetuned-rvlcdip"   # 768-d hidden size
processor = AutoImageProcessor.from_pretrained(MODEL_ID)
model      = AutoModel.from_pretrained(MODEL_ID)    # no classification head
device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device).eval()

# ----------------------------------------------------------------------
# 2.  Helper: render 1st page as RGB PIL image (use your tableai utils if preferred)
# ----------------------------------------------------------------------
def first_page_image(pdf_path, zoom=2):
    doc   = fitz.open(pdf_path)
    page  = doc.load_page(0)
    pix   = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom))
    return Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")

# ----------------------------------------------------------------------
# 3.  Extract CLS embeddings with DiT
# ----------------------------------------------------------------------
def dit_embed(image: Image.Image) -> np.ndarray:
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(**inputs).last_hidden_state       # [1, seq_len, 768]
    return out[:, 0, :].cpu().numpy().squeeze()       # CLS token (1×768 → 768)

# ----------------------------------------------------------------------
# 4.  Main work-flow
# ----------------------------------------------------------------------
# Common format that ends with '\n'  -> docker logs show each refresh
LINE_BAR = "{l_bar}{bar}| {n_fmt}/{total_fmt}  " \
           "{elapsed}<{remaining}, {rate_fmt}\n"

def cluster_dir(pdf_dir: str, out_yaml: str, min_cluster=4):
    paths, feats = [], []

    # tqdm now prints a \n per refresh; we reduce refresh rate with mininterval
    line_tqdm = partial(
        tqdm,
        file=sys.stdout,                      # stdout is the tty we allocated
        bar_format=LINE_BAR,
        mininterval=1.0,                      # refresh every second
        disable=not sys.stdout.isatty()       # still auto‑disable in CI
    )

    for fn in line_tqdm(sorted(os.listdir(pdf_dir)), desc="PDFs", unit="pdf"):
        if fn.lower().endswith(".pdf"):
            img   = first_page_image(os.path.join(pdf_dir, fn))
            vec   = dit_embed(img)
            paths.append(os.path.join(pdf_dir, fn))
            feats.append(vec)

    feats = np.vstack(feats)

    # 4a Dim-reduction (nice to visualise later)
    reducer  = umap.UMAP(n_neighbors=15, min_dist=0.05, random_state=42)
    emb2d    = reducer.fit_transform(feats)

    # 4b HDBSCAN clustering (density-based; no k required)
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster, metric="euclidean")
    labels    = clusterer.fit_predict(feats)          # −1 means “noise/outlier”

    # 4c YAML dump
    docs = [
        dict(file=path, cluster=int(lbl), umap_x=float(x), umap_y=float(y))
        for path, lbl, (x, y) in zip(paths, labels, emb2d)
    ]
    yaml.safe_dump({"documents": docs}, open(out_yaml, "w"))
    print(f"Wrote {out_yaml}")

    return emb2d, labels
