import asyncio
import time
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional
from app.core.config import settings
from app.core.errors import ChatSessionNotFoundError, SIFTException
from app.core.logging import logger
from app.rag.generation.base import BaseLLMProvider, LLMGenerationResult
from app.rag.generation.groq_provider import GroqProvider
from app.rag.generation.mock_provider import MockLLMProvider
from app.rag.generation.prompt_builder import PromptBuilder
from app.rag.retrieval.base import BaseRetriever
from app.rag.retrieval.retriever import semantic_retriever
from app.repositories.chat_repository import ChatRepository, chat_repository
from app.repositories.document_repository import DocumentRepository, document_repository
from app.schemas.chat import (
    ChatMessageResponse,
    ChatRequest,
    ChatResponse,
    ChatSessionDetail,
    ChatSessionSummary,
    CitationSchema,
)


class ChatService:
    """
    RAG conversation service uniting Phase 5 ChromaDB semantic retrieval,
    strict grounded prompt assembly, LLM generation, and SQLite conversational persistence.
    """

    def __init__(
        self,
        retriever_instance: BaseRetriever = semantic_retriever,
        chat_repo: ChatRepository = chat_repository,
        doc_repo: DocumentRepository = document_repository,
        provider: Optional[BaseLLMProvider] = None,
    ):
        self.retriever = retriever_instance
        self.chat_repo = chat_repo
        self.doc_repo = doc_repo
        self._provider = provider

    @property
    def provider(self) -> BaseLLMProvider:
        if self._provider is not None:
            return self._provider
        # Default initialization based on environment
        if settings.LLM_PROVIDER.lower() == "mock":
            return MockLLMProvider()
        return GroqProvider()

    @provider.setter
    def provider(self, val: BaseLLMProvider) -> None:
        self._provider = val

    async def generate_chat_response(self, request: ChatRequest) -> ChatResponse:
        start_time = time.perf_counter()
        query = request.effective_query

        if not query:
            raise SIFTException("Message or query cannot be empty.", status_code=400)

        # 1. Resolve or create chat session
        session_id = request.session_id
        if session_id:
            session = self.chat_repo.get_session(session_id)
            if not session:
                session = self.chat_repo.create_session(session_id=session_id, title=query[:40])
        else:
            session = self.chat_repo.create_session(title=query[:40])
        session_id = session["session_id"]

        # 2. Persist user turn
        self.chat_repo.add_message(session_id=session_id, role="user", content=query)

        # 3. Retrieve relevant chunks via Phase 5 ChromaDB semantic retrieval
        top_k = max(1, min(20, request.top_k or 5))
        retrieval_start = time.perf_counter()

        raw_passages = await asyncio.to_thread(
            self.retriever.retrieve,
            query=query,
            document_ids=request.document_ids,
            top_k=top_k,
            mode=request.retrieval_mode,
        )
        retrieval_latency_ms = (time.perf_counter() - retrieval_start) * 1000

        # 4. Construct verified citations linked strictly to retrieved passages
        citations: List[CitationSchema] = []
        for idx, p in enumerate(raw_passages, start=1):
            page_start = p.get("page_start")
            page_end = p.get("page_end")
            page_num = page_start if page_start is not None else None
            dist = float(p.get("distance", 0.0))
            sim = float(p.get("similarity_score", 0.0))
            cid = p.get("chunk_id", f"chk_{idx}")
            did = p.get("document_id", "")
            s_name = p.get("source_filename", "Document")

            citations.append(
                CitationSchema(
                    id=f"cit-{cid}-{idx}",
                    document_id=did,
                    chunk_id=cid,
                    chunk_index=p.get("chunk_index", 0),
                    source_filename=s_name,
                    page_start=page_start,
                    page_end=page_end,
                    passage=p.get("content", ""),
                    distance=dist,
                    similarity_score=sim,
                    docId=did,
                    docName=s_name,
                    pageNumber=page_num,
                    chunkId=cid,
                    similarityScore=sim,
                    sectionTitle=f"Passage [{idx}]" + (f" (Page {page_num})" if page_num else ""),
                )
            )

        # 5. Grounded Generation
        gen_res: Optional[LLMGenerationResult] = None
        if not raw_passages:
            answer_text = "Based on the provided documents, there is insufficient evidence to answer your question."
            provider_name = self.provider.provider_name
            model_name = self.provider.model_name
        else:
            # Fetch bounded conversation history
            if request.history:
                history_turns = [{"role": m.role, "content": m.content} for m in request.history]
            else:
                raw_history = self.chat_repo.get_recent_history(
                    session_id=session_id, limit=settings.CHAT_HISTORY_LIMIT + 1
                )
                # Omit the last message if it's the current user query to prevent duplication
                if raw_history and raw_history[-1]["content"] == query:
                    history_turns = raw_history[:-1]
                else:
                    history_turns = raw_history

            prompt = PromptBuilder.build_grounded_prompt(
                query=query,
                retrieved_passages=raw_passages,
            )

            gen_res = await self.provider.generate(
                prompt=prompt,
                system_instruction=PromptBuilder.SYSTEM_PROMPT,
                conversation_history=history_turns,
            )
            answer_text = gen_res.text
            provider_name = gen_res.provider
            model_name = gen_res.model

        # 6. Persist assistant turn with citations
        msg_record = self.chat_repo.add_message(
            session_id=session_id,
            role="assistant",
            content=answer_text,
            citations=[c.model_dump() for c in citations],
            metadata={
                "provider": provider_name,
                "model": model_name,
                "retrieval_count": len(raw_passages),
            },
        )

        total_latency_ms = (time.perf_counter() - start_time) * 1000
        avg_confidence = round(
            sum(c.similarity_score for c in citations) / len(citations), 2
        ) if citations else 0.85

        logger.info(
            f"Completed RAG chat interaction ({provider_name}/{model_name}) "
            f"in {total_latency_ms:.1f}ms (citations: {len(citations)})"
        )

        return ChatResponse(
            id=msg_record["message_id"],
            session_id=session_id,
            role="assistant",
            answer=answer_text,
            content=answer_text,
            timestamp=datetime.now().strftime("%I:%M %p"),
            citations=citations,
            retrieved_chunks=raw_passages,
            provider=provider_name,
            model=model_name,
            retrieval_metadata={
                "latency_ms": round(retrieval_latency_ms, 2),
                "chunks_retrieved": len(raw_passages),
                "metric": "cosine_distance",
            },
            generation_metadata={
                "latency_ms": gen_res.latency_ms if gen_res else 0.0,
                "tokens_used": gen_res.tokens_used if gen_res else 0,
                "finish_reason": gen_res.finish_reason if gen_res else "insufficient_evidence",
            },
            confidence_score=avg_confidence,
            latency_ms=round(total_latency_ms, 1),
            tokens_processed=(gen_res.tokens_used if gen_res and gen_res.tokens_used else 0) + len(query.split()),
            retrieval_mode=request.retrieval_mode,
        )

    def list_sessions(self, limit: int = 50) -> List[ChatSessionSummary]:
        sessions = self.chat_repo.list_sessions(limit=limit)
        return [
            ChatSessionSummary(
                session_id=s["session_id"],
                title=s["title"],
                created_at=s["created_at"],
                updated_at=s["updated_at"],
                message_count=s["message_count"],
            )
            for s in sessions
        ]

    def get_session_detail(self, session_id: str) -> ChatSessionDetail:
        session = self.chat_repo.get_session(session_id)
        if not session:
            raise ChatSessionNotFoundError(session_id)

        messages = self.chat_repo.get_messages(session_id)
        parsed_messages: List[ChatMessageResponse] = []
        for m in messages:
            cits = []
            for c in m.get("citations", []):
                try:
                    cits.append(CitationSchema(**c))
                except Exception:
                    pass
            parsed_messages.append(
                ChatMessageResponse(
                    message_id=m["message_id"],
                    session_id=m["session_id"],
                    role=m["role"],
                    content=m["content"],
                    citations=cits,
                    created_at=m["created_at"],
                )
            )

        return ChatSessionDetail(
            session_id=session["session_id"],
            title=session["title"],
            created_at=session["created_at"],
            updated_at=session["updated_at"],
            messages=parsed_messages,
        )

    def delete_session(self, session_id: str) -> bool:
        session = self.chat_repo.get_session(session_id)
        if not session:
            raise ChatSessionNotFoundError(session_id)
        return self.chat_repo.delete_session(session_id)


# Default singleton instance
chat_service = ChatService()
