
from fastapi import (
    APIRouter,
    Query, 
    Depends,
    HTTPException, 
    Body
)
import tempfile
from fastapi.responses import FileResponse

import json
from typing import List, Any
from pydantic import BaseModel
from typing import Optional, Dict, Any, Union
from pathlib import Path
import fitz


### Application imports ###
# from api.extract.base import BaseNode
from api.service.dependencies import get_db, ensure_initialized
from backend.models.backend import ClassificationLabel
### ------------------- ###

router = APIRouter()


# /Users/johnschlafly/Documents/pdfs/tableai/tableai/.synced/dropbox/pdf/id_q96KjlOoc_kAAAAAAAAA-A.pdf
# /Users/johnschlafly/Documents/pdfs/tableai/tableai/.synced/dropbox/pdf

@router.get("/classify/sample_preview/{file_id}")
def get_pdf_preview(file_id: str):
    # Construct expected file path
    pdf_dir = Path(__file__).parent.parent.parent.parent / "tableai" / ".synced" / "dropbox" / "pdf"
    possible_paths = list(pdf_dir.rglob(f"{file_id}.*"))  # Handle .pdf or other extensions

    print(possible_paths)
    print(pdf_dir)
    if not possible_paths:
        raise HTTPException(status_code=404, detail="PDF not found")

    pdf_path = possible_paths[0]

    # Convert first page to PNG
    with fitz.open(pdf_path) as doc:
        if doc.page_count == 0:
            raise HTTPException(status_code=400, detail="Empty PDF")
        page = doc[0]
        pix = page.get_pixmap(dpi=200)

        # Save to temp PNG file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        pix.save(temp_file.name)

    return FileResponse(temp_file.name, media_type="image/png")


@router.post("/classify/labels")
def save_classification_labels(
    payload: dict = Body(...),
    api_service: 'APIService' = Depends(ensure_initialized)
    ):
    """
    Expects a payload like:
    {
        "creator|OpenTextExstreamVersion22...": "OpenText Statements",
        "creator|RicohAmericasCorporation...": "Ricoh Printouts"
    }
    """
    db = api_service.db
    for classification, label in payload.items():
        db.run_op(
            ClassificationLabel,
            operation="merge",
            data={
                "classification": classification,
                "label": label
            }
        )
    return {"status": "saved", "count": len(payload)}