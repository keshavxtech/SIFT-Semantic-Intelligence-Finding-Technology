from typing import List
from fastapi import APIRouter, Query, status
from app.schemas.chat import (
    ChatRequest,
    ChatResponse,
    ChatSessionDetail,
    ChatSessionSummary,
)
from app.services.chat_service import chat_service

router = APIRouter(prefix="/chat", tags=["Chat"])


@router.post("", response_model=ChatResponse)
async def chat_interaction(request: ChatRequest) -> ChatResponse:
    """
    RAG conversation endpoint.
    Retrieves top-k passages via Phase 5 ChromaDB and synthesizes a grounded answer with citations.
    """
    return await chat_service.generate_chat_response(request)


@router.get("/sessions", response_model=List[ChatSessionSummary])
async def list_chat_sessions(
    limit: int = Query(50, ge=1, le=100, description="Max sessions to list"),
) -> List[ChatSessionSummary]:
    """Lists available chat conversation sessions."""
    return chat_service.list_sessions(limit=limit)


@router.get("/sessions/{session_id}", response_model=ChatSessionDetail)
async def get_chat_session(session_id: str) -> ChatSessionDetail:
    """Retrieves full conversation turns and citations for a specific chat session."""
    return chat_service.get_session_detail(session_id)


@router.delete("/sessions/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_chat_session(session_id: str) -> None:
    """Permanently deletes a chat session and all its associated conversation turns."""
    chat_service.delete_session(session_id)
