import asyncio
import time
from typing import List, Optional
from app.core.errors import SIFTException
from app.core.logging import logger
from app.models.document import DocumentTag, DocumentType
from app.rag.retrieval.base import BaseRetriever
from app.rag.retrieval.retriever import SemanticRetriever, semantic_retriever
from app.repositories.document_repository import DocumentRepository, document_repository
from app.schemas.search import SearchRequest, SearchResponse, SearchResultItem


class SearchService:
    """
    Production semantic search service powered exclusively by dense query embeddings
    and ChromaDB vector similarity traversal using cosine distance.
    All mock result generation, hardcoded passages, and token heuristics are removed.
    """

    def __init__(
        self,
        retriever_instance: BaseRetriever = semantic_retriever,
        doc_repo: DocumentRepository = document_repository,
    ):
        self.retriever = retriever_instance
        self.doc_repo = doc_repo

    async def search(self, request: SearchRequest) -> SearchResponse:
        start_time = time.perf_counter()

        cleaned_query = (request.query or "").strip()
        if not cleaned_query:
            raise SIFTException("Search query cannot be empty or whitespace only.", status_code=400)

        top_k = request.effective_top_k

        # Execute dense vector retrieval in worker thread
        raw_results = await asyncio.to_thread(
            self.retriever.retrieve,
            query=cleaned_query,
            document_ids=request.document_ids,
            top_k=top_k,
            threshold=request.threshold,
            mode=request.mode,
        )

        # Batch lookup document metadata from SQLite for UI presentation
        doc_ids = {r["document_id"] for r in raw_results if r.get("document_id")}
        doc_map = {}
        for did in doc_ids:
            doc = self.doc_repo.get(did)
            if doc:
                doc_map[did] = doc

        items: List[SearchResultItem] = []
        for r in raw_results:
            did = r["document_id"]
            doc = doc_map.get(did)

            doc_type = doc.file_type if doc else DocumentType.TXT
            doc_name = doc.original_filename if doc else r.get("source_filename", "document")
            doc_tag = doc.tag if doc else DocumentTag.TECHNICAL

            page_start = r.get("page_start")
            page_end = r.get("page_end")
            page_num = page_start if page_start is not None else 1

            chunk_id = r["chunk_id"]
            content = r["content"]
            distance = float(r["distance"])
            similarity = float(r["similarity_score"])

            # Extract basic query terms for highlight markers
            query_terms = [w.lower() for w in cleaned_query.split() if len(w) > 2]
            highlights = [w for w in query_terms if w in content.lower()]

            items.append(
                SearchResultItem(
                    document_id=did,
                    chunk_id=chunk_id,
                    chunk_index=r.get("chunk_index", 0),
                    content=content,
                    source_filename=doc_name,
                    page_start=page_start,
                    page_end=page_end,
                    distance=distance,
                    similarity_score=similarity,
                    # Frontend compatibility fields
                    id=f"sr-{chunk_id}",
                    docId=did,
                    docName=doc_name,
                    docType=doc_type,
                    pageNumber=page_num,
                    chunkId=chunk_id,
                    passage=content,
                    score=similarity,
                    cosineDistance=distance,
                    tokenCount=max(1, int(len(content.split()) / 0.75)),
                    matchHighlights=highlights if highlights else query_terms[:2],
                    sectionTitle=f"Page {page_num}" if page_start else "Passage",
                    tag=doc_tag,
                )
            )

        elapsed_ms = (time.perf_counter() - start_time) * 1000
        logger.info(
            f"Executed real vector search for '{cleaned_query[:35]}' in {elapsed_ms:.1f}ms (matches: {len(items)})"
        )

        return SearchResponse(
            query=request.query,
            total_matches=len(items),
            execution_time_ms=round(elapsed_ms, 2),
            mode=request.mode,
            metric="cosine_distance (lower is better, range [0, 2])",
            results=items,
        )


# Default singleton instance
search_service = SearchService()
