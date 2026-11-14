from typing import Any, Dict, List, Optional
from app.core.errors import SIFTException
from app.core.logging import logger
from app.rag.embeddings.base import BaseEmbeddingProvider
from app.rag.retrieval.base import BaseRetriever, BaseVectorStore
from app.rag.retrieval.vector_store import vector_store


class SemanticRetriever(BaseRetriever):
    """
    Semantic passage retriever that uses dense query embeddings and
    ChromaDB cosine distance nearest-neighbor traversal.
    """

    def __init__(
        self,
        vector_store_instance: BaseVectorStore = vector_store,
        provider: Optional[BaseEmbeddingProvider] = None,
    ):
        self.vector_store = vector_store_instance
        self._custom_provider = provider

    @property
    def provider(self) -> BaseEmbeddingProvider:
        if self._custom_provider:
            return self._custom_provider
        from app.rag.embeddings.embedding_service import embedding_service
        return embedding_service.provider

    def retrieve(
        self,
        query: str,
        document_ids: Optional[List[str]] = None,
        top_k: int = 5,
        threshold: Optional[float] = None,
        mode: str = "dense",
    ) -> List[Dict[str, Any]]:
        """
        Embeds query text, validates vector, and retrieves top-k nearest chunks from ChromaDB.
        """
        cleaned_query = (query or "").strip()
        if not cleaned_query:
            return []

        top_k = max(1, min(20, top_k))

        # 1. Generate query embedding using Phase 4 embedding provider
        query_vector = self.provider.embed_text(cleaned_query)

        # 2. Verify dimension
        if not self.provider.validate_vector(query_vector):
            raise SIFTException(
                f"Generated query vector failed validation (dimension {len(query_vector)}, expected {self.provider.dimension})",
                status_code=400,
            )

        # 3. Query ChromaDB vector store
        results = self.vector_store.query(
            query_embedding=query_vector,
            top_k=top_k,
            document_ids=document_ids,
        )

        # 4. Filter by threshold if requested
        if threshold is not None:
            results = [r for r in results if r["similarity_score"] >= threshold]

        logger.info(
            f"Semantic retrieval for query '{cleaned_query[:40]}' returned {len(results)} matches (top_k={top_k})"
        )
        return results


# Default singleton instance
semantic_retriever = SemanticRetriever()
