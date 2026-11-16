import math
from typing import Any, Dict, List, Optional
import httpx

from app.core.config import settings
from app.core.errors import DimensionMismatchError, SIFTException, VectorStoreNotConfiguredError
from app.core.logging import logger
from app.rag.retrieval.base import BaseVectorStore


class PineconeRestClient:
    """
    Lightweight REST client for Pinecone Data Plane API using httpx.
    Allows Pinecone vector storage operations without requiring heavy client dependencies.
    """

    def __init__(self, api_key: str, host: str, timeout_seconds: float = 30.0):
        self.api_key = api_key
        # Clean host URL (ensure https:// prefix and no trailing slash)
        raw_host = host.strip().rstrip("/")
        if not raw_host.startswith("http://") and not raw_host.startswith("https://"):
            raw_host = f"https://{raw_host}"
        self.host = raw_host
        self.timeout = timeout_seconds

    @property
    def headers(self) -> Dict[str, str]:
        return {
            "Api-Key": self.api_key,
            "Content-Type": "application/json",
            "X-Pinecone-API-Version": "2024-07",
        }

    def upsert(self, vectors: List[Dict[str, Any]], namespace: Optional[str] = None) -> Dict[str, Any]:
        url = f"{self.host}/vectors/upsert"
        payload: Dict[str, Any] = {"vectors": vectors}
        if namespace:
            payload["namespace"] = namespace

        with httpx.Client(timeout=self.timeout) as client:
            resp = client.post(url, json=payload, headers=self.headers)
            if resp.status_code != 200:
                raise SIFTException(f"Pinecone upsert failed ({resp.status_code}): {resp.text[:200]}", status_code=502)
            return resp.json()

    def query(
        self,
        vector: List[float],
        top_k: int = 5,
        filter_dict: Optional[Dict[str, Any]] = None,
        include_metadata: bool = True,
        namespace: Optional[str] = None,
    ) -> Dict[str, Any]:
        url = f"{self.host}/query"
        payload: Dict[str, Any] = {
            "vector": vector,
            "topK": top_k,
            "includeMetadata": include_metadata,
        }
        if filter_dict:
            payload["filter"] = filter_dict
        if namespace:
            payload["namespace"] = namespace

        with httpx.Client(timeout=self.timeout) as client:
            resp = client.post(url, json=payload, headers=self.headers)
            if resp.status_code != 200:
                raise SIFTException(f"Pinecone query failed ({resp.status_code}): {resp.text[:200]}", status_code=502)
            return resp.json()

    def delete(
        self,
        ids: Optional[List[str]] = None,
        filter_dict: Optional[Dict[str, Any]] = None,
        delete_all: bool = False,
        namespace: Optional[str] = None,
    ) -> Dict[str, Any]:
        url = f"{self.host}/vectors/delete"
        payload: Dict[str, Any] = {}
        if ids:
            payload["ids"] = ids
        if filter_dict:
            payload["filter"] = filter_dict
        if delete_all:
            payload["deleteAll"] = True
        if namespace:
            payload["namespace"] = namespace

        with httpx.Client(timeout=self.timeout) as client:
            resp = client.post(url, json=payload, headers=self.headers)
            if resp.status_code != 200:
                raise SIFTException(f"Pinecone delete failed ({resp.status_code}): {resp.text[:200]}", status_code=502)
            return resp.json()

    def describe_index_stats(self, filter_dict: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        url = f"{self.host}/describe_index_stats"
        payload: Dict[str, Any] = {}
        if filter_dict:
            payload["filter"] = filter_dict

        with httpx.Client(timeout=self.timeout) as client:
            resp = client.post(url, json=payload, headers=self.headers)
            if resp.status_code != 200:
                raise SIFTException(f"Pinecone describe_index_stats failed ({resp.status_code}): {resp.text[:200]}", status_code=502)
            return resp.json()


class PineconeVectorStore(BaseVectorStore):
    """
    Pinecone Cloud vector store adapter implementing BaseVectorStore.
    Supports 384-dimensional dense vectors with cosine similarity matching,
    metadata filtering by document_id, and automatic fallback between official SDK and lightweight REST API.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        index_name: Optional[str] = None,
        namespace: Optional[str] = None,
        host: Optional[str] = None,
        expected_dimension: int = 384,
        client: Optional[Any] = None,
    ):
        self.api_key = api_key or settings.PINECONE_API_KEY
        self.index_name = index_name or settings.PINECONE_INDEX_NAME
        self.namespace = namespace or settings.PINECONE_NAMESPACE
        self.host = host or settings.PINECONE_HOST
        self.expected_dimension = expected_dimension
        self._client = client

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client

        if not self.api_key or not self.api_key.strip():
            raise VectorStoreNotConfiguredError("PINECONE_API_KEY is not configured.")

        # Try official Pinecone SDK if installed
        try:
            from pinecone import Pinecone
            pc = Pinecone(api_key=self.api_key)
            if self.host:
                self._client = pc.Index(name=self.index_name, host=self.host)
            else:
                self._client = pc.Index(name=self.index_name)
            logger.info(f"Connected to Pinecone index '{self.index_name}' via official SDK")
            return self._client
        except (ImportError, Exception) as sdk_err:
            logger.info(f"Pinecone SDK unavailable or host required ({sdk_err}), using lightweight REST client")

        # Fallback to REST client
        if not self.host:
            raise VectorStoreNotConfiguredError(
                "PINECONE_HOST (e.g. 'https://sift-index-xxxx.svc.pinecone.io') is required when using Pinecone REST adapter."
            )
        self._client = PineconeRestClient(api_key=self.api_key, host=self.host)
        return self._client

    def upsert_chunks(
        self,
        document_id: str,
        chunk_ids: List[str],
        embeddings: List[List[float]],
        documents: List[str],
        metadatas: List[Dict[str, Any]],
    ) -> int:
        if not chunk_ids:
            return 0

        # Validate dimensions
        for idx, emb in enumerate(embeddings):
            if len(emb) != self.expected_dimension:
                raise DimensionMismatchError(
                    expected=self.expected_dimension,
                    actual=len(emb),
                )

        client = self._get_client()
        vectors_to_upsert: List[Dict[str, Any]] = []

        for cid, emb, text, meta in zip(chunk_ids, embeddings, documents, metadatas):
            sanitized_meta = dict(meta) if meta else {}
            sanitized_meta["document_id"] = document_id
            sanitized_meta["content"] = text
            sanitized_meta["chunk_id"] = cid

            # Ensure numeric values are valid types for Pinecone metadata
            for k, v in list(sanitized_meta.items()):
                if v is None:
                    sanitized_meta[k] = ""
                elif not isinstance(v, (str, int, float, bool, list)):
                    sanitized_meta[k] = str(v)

            vectors_to_upsert.append({
                "id": cid,
                "values": emb,
                "metadata": sanitized_meta,
            })

        # Batch upsert (max 100 per call for reliability)
        batch_size = 100
        for i in range(0, len(vectors_to_upsert), batch_size):
            batch = vectors_to_upsert[i : i + batch_size]
            if hasattr(client, "upsert"):
                client.upsert(vectors=batch, namespace=self.namespace)
            else:
                client.upsert(batch, namespace=self.namespace)

        logger.info(f"Successfully upserted {len(vectors_to_upsert)} chunks to Pinecone for doc '{document_id}'")
        return len(vectors_to_upsert)

    def delete_document(self, document_id: str) -> int:
        client = self._get_client()
        filter_dict = {"document_id": {"$eq": document_id}}
        try:
            if hasattr(client, "delete"):
                client.delete(filter=filter_dict, namespace=self.namespace)
            logger.info(f"Deleted vectors for document '{document_id}' from Pinecone")
            return 1
        except Exception as e:
            logger.warning(f"Error deleting document '{document_id}' from Pinecone: {e}")
            return 0

    def query(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        document_ids: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        if len(query_embedding) != self.expected_dimension:
            raise DimensionMismatchError(
                expected=self.expected_dimension,
                actual=len(query_embedding),
            )

        client = self._get_client()
        filter_dict: Optional[Dict[str, Any]] = None
        if document_ids and len(document_ids) > 0:
            if len(document_ids) == 1:
                filter_dict = {"document_id": {"$eq": document_ids[0]}}
            else:
                filter_dict = {"document_id": {"$in": document_ids}}

        resp = client.query(
            vector=query_embedding,
            top_k=top_k,
            filter=filter_dict,
            include_metadata=True,
            namespace=self.namespace,
        )

        matches = resp.get("matches", []) if isinstance(resp, dict) else getattr(resp, "matches", [])
        results: List[Dict[str, Any]] = []

        for m in matches:
            m_dict = m if isinstance(m, dict) else m.to_dict() if hasattr(m, "to_dict") else vars(m)
            cid = m_dict.get("id", "")
            score = float(m_dict.get("score", 0.0))
            meta = m_dict.get("metadata", {}) or {}
            did = meta.get("document_id", "")
            content = meta.get("content", "")
            s_name = meta.get("source_filename", "")
            p_start = meta.get("page_start")
            p_end = meta.get("page_end")
            c_idx = meta.get("chunk_index", 0)

            # Cosine distance in Pinecone is 1 - score (for normalized vectors)
            distance = max(0.0, 1.0 - score)

            results.append({
                "id": cid,
                "chunk_id": cid,
                "document_id": did,
                "content": content,
                "passage": content,
                "source_filename": s_name,
                "page_start": p_start,
                "page_end": p_end,
                "chunk_index": c_idx,
                "distance": distance,
                "similarity_score": score,
                "score": score,
                "metadata": meta,
                "docId": did,
                "docName": s_name,
                "pageNumber": p_start,
                "chunkId": cid,
                "similarityScore": score,
                "cosineDistance": distance,
            })

        return results

    def count(self) -> int:
        client = self._get_client()
        try:
            stats = client.describe_index_stats()
            stats_dict = stats if isinstance(stats, dict) else stats.to_dict() if hasattr(stats, "to_dict") else vars(stats)
            if self.namespace and "namespaces" in stats_dict:
                ns_stats = stats_dict["namespaces"].get(self.namespace, {})
                return ns_stats.get("vector_count", 0)
            return stats_dict.get("total_vector_count", 0)
        except Exception as e:
            logger.warning(f"Error obtaining Pinecone stats: {e}")
            return 0

    def get_document_chunk_count(self, document_id: str) -> int:
        client = self._get_client()
        try:
            # Query top 1000 with dummy zero vector to count matches for document_id
            zero_vec = [0.0] * self.expected_dimension
            resp = client.query(
                vector=zero_vec,
                top_k=1000,
                filter={"document_id": {"$eq": document_id}},
                include_metadata=False,
                namespace=self.namespace,
            )
            matches = resp.get("matches", []) if isinstance(resp, dict) else getattr(resp, "matches", [])
            return len(matches)
        except Exception as e:
            logger.warning(f"Error getting chunk count for document '{document_id}' in Pinecone: {e}")
            return 0

    def is_document_indexed(self, document_id: str) -> bool:
        return self.get_document_chunk_count(document_id) > 0
