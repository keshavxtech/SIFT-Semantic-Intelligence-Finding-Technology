from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class BaseVectorStore(ABC):
    """Abstract interface for dense vector persistence and similarity retrieval."""

    @abstractmethod
    def upsert_chunks(
        self,
        document_id: str,
        chunk_ids: List[str],
        embeddings: List[List[float]],
        documents: List[str],
        metadatas: List[Dict[str, Any]],
    ) -> int:
        """Inserts or updates vector records in the store. Returns count of inserted records."""
        pass

    @abstractmethod
    def delete_document(self, document_id: str) -> int:
        """Removes all vectors associated with a document_id."""
        pass

    @abstractmethod
    def query(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        document_ids: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Queries the vector store for nearest neighbours by cosine distance."""
        pass

    @abstractmethod
    def count(self) -> int:
        """Returns the total number of vectors in the store."""
        pass

    @abstractmethod
    def get_document_chunk_count(self, document_id: str) -> int:
        """Returns the number of vectors stored for a specific document_id."""
        pass

    @abstractmethod
    def is_document_indexed(self, document_id: str) -> bool:
        """Checks if at least one chunk vector exists for the document_id."""
        pass


class BaseRetriever(ABC):
    """Abstract interface for hybrid neural + sparse retrieval."""

    @abstractmethod
    def retrieve(
        self,
        query: str,
        document_ids: Optional[List[str]] = None,
        top_k: int = 5,
        threshold: Optional[float] = None,
        mode: str = "dense",
    ) -> List[Dict[str, Any]]:
        pass
