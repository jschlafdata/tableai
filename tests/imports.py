from tableai_plugins import get_plugin, list_plugins

print(list_plugins())  # {'yolo': 'tableai_plugins.yolo:YOLOPlugin', 'dit': 'tableai_plugins.dit:DiTPlugin'}

# YOLO: detect tables on an image (np.ndarray or PIL.Image)
YOLO = get_plugin("yolo")
yolo = YOLO(model_name="keremberke")  # or "foduucom", "doclaynet"
import numpy as np
dummy = np.zeros((640, 640, 3), dtype=np.uint8)
yres = yolo.detect(dummy, page_num=0)
print(yres.data)

# DiT: embed an image or PDF, optionally cluster a directory of PDFs
DiT = get_plugin("dit")
dit = DiT()  # optionally DiT(model_id="microsoft/dit-base-finetuned-rvlcdip")
vec = dit.embed("some.pdf")
docs, emb2d, labels = dit.cluster_pdfs("/path/to/pdf_dir", out_yaml="clusters.yaml")