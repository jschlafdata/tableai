import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from integrations.dropbox.auth import DropboxAuth

app = FastAPI()

# allow React dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)

# initialize Dropbox client
auth = DropboxAuth()
dbx = auth.get_client()


@app.get("/folders/{path:path}")
async def list_folder(path: str):
    """
    List contents of a Dropbox “folder” at the given path.
    Dropbox uses ‘’ or ‘/’ for the root.
    """
    # normalize root
    if path in ("", "/", "root"):
        db_path = ""
    else:
        db_path = f"/{path.lstrip('/')}"
    try:
        res = dbx.files_list_folder(db_path)
        return [
            {
                "name": e.name,
                "path_lower": e.path_lower,
                "type": "folder" if isinstance(e, 
                    __import__("dropbox").files.FolderMetadata) else "file"
            }
            for e in res.entries
        ]
    except Exception as e:
        raise HTTPException(500, str(e))