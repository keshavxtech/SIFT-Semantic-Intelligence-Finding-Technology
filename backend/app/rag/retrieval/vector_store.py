import math
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional
from app.core.config import settings
from app.core.errors import DimensionMismatchError, IndexingError, VectorStoreNotConfiguredError
from app.core.logging import logger
from app.rag.retrieval.base import BaseVectorStore


class ChromaVectorStore(BaseVectorStore):
    """
    Persistent ChromaDB vector store configured with HNSW cosine distance metric.
    Stores and indexes 384-dimensional dense vectors with chunk metadata.
    """

    def __init__(
        self,
        persist_directory: Optional[Path] = None,
        collection_name: Optional[str] = None,
        expected_dimension: int = 384,
    ):
        self.persist_directory = persist_directory or settings.vector_db_path
        self.collection_name = collection_name or settings.CHROMA_COLLECTION_NAME
        self.expected_dimension = expected_dimension
        self._client: Optional[Any] = None
        self._collection = None
        self._lock = threading.RLock()

    def _get_client(self) -> Any:
        if self._client is None:
            with self._lock:
                if self._client is None:
                    try:
                        import chromadb
                    except ImportError:
                        raise VectorStoreNotConfiguredError(
                            "ChromaDB is not installed in this environment. "
                            "In production, configure VECTOR_STORE_PROVIDER='pinecone'."
                        )
                    self.persist_directory.mkdir(parents=True, exist_ok=True)
                    logger.info(
                        f"Initializing Persistent ChromaDB client at '{self.persist_directory.resolve()}'"
                    )
                    self._client = chromadb.PersistentClient(path=str(self.persist_directory.resolve()))
        return self._client

    def _get_collection(self):
        if self._collection is None:
            with self._lock:
                if self._collection is None:
                    client = self._get_client()
                    logger.info(
                        f"Connecting to ChromaDB collection '{self.collection_name}' (space=cosine)"
                    )
                    self._collection = client.get_or_create_collection(
                        name=self.collection_name,
                        metadata={"hnsw:space": "cosine"},
                    )
        return self._collection

    @property
    def collection(self):
        return self._get_collection()

    def _sanitize_metadata(self, meta: Dict[str, Any]) -> Dict[str, Any]:
        """Ensures all metadata fields are valid primitive types for ChromaDB."""
        sanitized = {}
        for k, v in meta.items():
            if v is None:
                if "page" in k.lower():
                    sanitized[k] = -1
                else:
                    sanitized[k] = ""
            elif isinstance(v, (str, int, float, bool)):
                sanitized[k] = v
            else:
                sanitized[k] = str(v)
        return sanitized

    def _validate_vector(self, vec: List[float], chunk_id: str = "") -> None:
        if not isinstance(vec, list) or len(vec) != self.expected_dimension:
            raise DimensionMismatchError(
                expected=self.expected_dimension,
                actual=len(vec) if isinstance(vec, list) else 0,
            )
        for val in vec:
            if math.isnan(val) or math.isinf(val):
                raise IndexingError(
                    f"Vector contains invalid float value (NaN or Inf) in chunk '{chunk_id}'"
                )

    def upsert_chunks(
        self,
        document_id: str,
        chunk_ids: List[str],
        embeddings: List[List[float]],
        documents: List[str],
        metadatas: List[Dict[str, Any]],
    ) -> int:
        """
        Idempotently inserts or updates vectors for a document in ChromaDB.
        Removes existing vectors for this document first to prevent dangling chunks on re-indexing.
        """
        if not chunk_ids:
            return 0

        if len(chunk_ids) != len(embeddings) or len(chunk_ids) != len(documents) or len(chunk_ids) != len(metadatas):
            raise IndexingError(
                f"Mismatched input lengths for document '{document_id}': "
                f"{len(chunk_ids)} chunk_ids, {len(embeddings)} embeddings, "
                f"{len(documents)} documents, {len(metadatas)} metadatas."
            )

        # Validate all vectors
        for cid, vec in zip(chunk_ids, embeddings):
            self._validate_vector(vec, chunk_id=cid)

        # Sanitize metadata
        sanitized_metas = [self._sanitize_metadata(m) for m in metadatas]

        col = self.collection

        # Safe re-indexing: delete existing records for this document to avoid orphans
        self.delete_document(document_id)

        # Upsert into ChromaDB
        col.add(
            ids=chunk_ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=sanitized_metas,
        )

        logger.info(
            f"Successfully indexed {len(chunk_ids)} chunks for document '{document_id}' into ChromaDB collection '{self.collection_name}'"
        )
        return len(chunk_ids)

    def delete_document(self, document_id: str) -> int:
        """Removes all indexed vectors associated with a document_id."""
        col = self.collection
        try:
            if col.count() == 0:
                return 0
            existing = col.get(where={"document_id": document_id})
            ids = existing.get("ids", [])
            if ids:
                col.delete(ids=ids)
                logger.info(f"Deleted {len(ids)} vectors for document '{document_id}' from ChromaDB")
                return len(ids)
            return 0
        except Exception as e:
            logger.warning(f"ChromaDB delete query for document '{document_id}' encountered: {e}")
            return 0

    def query(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        document_ids: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Performs nearest-neighbor search by cosine distance.
        Applies document_id filter if provided.
        """
        self._validate_vector(query_embedding, chunk_id="query")

        # Clamp top_k between 1 and 20
        top_k = max(1, min(20, top_k))

        total_vectors = self.count()
        if total_vectors == 0:
            return []

        col = self.collection

        # Build where filter
        where_filter = None
        if document_ids is not None:
            clean_ids = [d.strip() for d in document_ids if d and d.strip()]
            if len(clean_ids) == 1:
                where_filter = {"document_id": clean_ids[0]}
            elif len(clean_ids) > 1:
                where_filter = {"document_id": {"$in": clean_ids}}
            elif len(document_ids) > 0 and len(clean_ids) == 0:
                # User passed explicit empty strings or non-existent ids
                return []

        # ChromaDB n_results cannot exceed total items in collection
        n_results = min(top_k, total_vectors)

        try:
            query_res = col.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                where=where_filter,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as e:
            logger.error(f"ChromaDB query execution failed: {e}")
            return []

        results: List[Dict[str, Any]] = []
        ids = query_res.get("ids", [[]])[0]
        docs = query_res.get("documents", [[]])[0]
        metas = query_res.get("metadatas", [[]])[0]
        distances = query_res.get("distances", [[]])[0]

        for chunk_id, content, meta, dist in zip(ids, docs, metas, distances):
            meta = meta or {}
            page_start = meta.get("page_start")
            if page_start == -1:
                page_start = None
            page_end = meta.get("page_end")
            if page_end == -1:
                page_end = None

            dist_val = float(dist)
            # Cosine similarity: 1.0 - distance
            sim_score = max(0.0, min(1.0, 1.0 - dist_val))

            results.append({
                "chunk_id": chunk_id,
                "document_id": str(meta.get("document_id", "")),
                "chunk_index": int(meta.get("chunk_index", 0)),
                "content": content or "",
                "source_filename": str(meta.get("source_filename", "")),
                "file_type": str(meta.get("file_type", "txt")),
                "page_start": page_start,
                "page_end": page_end,
                "embedding_model": str(meta.get("embedding_model", "")),
                "distance": dist_val,
                "similarity_score": round(sim_score, 4),
            })

        return results

    def count(self) -> int:
        col = self.collection
        return col.count()

    def get_document_chunk_count(self, document_id: str) -> int:
        col = self.collection
        try:
            res = col.get(where={"document_id": document_id})
            return len(res["ids"]) if res and "ids" in res else 0
        except Exception as e:
            logger.warning(f"Error checking chunk count for document '{document_id}': {e}")
            return 0

    def is_document_indexed(self, document_id: str) -> bool:
        return self.get_document_chunk_count(document_id) > 0


def get_vector_store(provider_type: Optional[str] = None) -> BaseVectorStore:
    """
    Factory resolving active vector store implementation based on configuration.
    Supports local persistent ChromaDB and cloud Pinecone vector store.
    """
    name = (provider_type or settings.VECTOR_STORE_PROVIDER).lower().strip()
    if name == "pinecone":
        from app.rag.retrieval.pinecone_vector_store import PineconeVectorStore
        return PineconeVectorStore()
    else:
        # Default local persistent ChromaDB
        return ChromaVectorStore()


# Default singleton instance
vector_store = get_vector_store()
