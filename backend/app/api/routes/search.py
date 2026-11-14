from fastapi import APIRouter
from app.schemas.search import SearchRequest, SearchResponse
from app.services.search_service import search_service

router = APIRouter(prefix="/search", tags=["Search"])


@router.post("", response_model=SearchResponse)
async def perform_search(request: SearchRequest) -> SearchResponse:
    return await search_service.search(request)
