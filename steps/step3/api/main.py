from fastapi import APIRouter

from api.routes import index
from api.routes import file
from api.routes import model

api_router = APIRouter()
api_router.include_router(index.router, tags=["index"])
api_router.include_router(file.router, prefix="/file", tags=["file"])
api_router.include_router(model.router, prefix="/model", tags=["model"])