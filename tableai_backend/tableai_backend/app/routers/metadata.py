from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

router = APIRouter(prefix="/metadata", tags=["metadata"])

@router.get("/openapi", include_in_schema=False, response_class=JSONResponse)
async def get_api_schema(request: Request):
    return request.app.openapi()