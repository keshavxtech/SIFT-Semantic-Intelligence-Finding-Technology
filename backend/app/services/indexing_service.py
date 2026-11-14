import asyncio
from typing import Optional
from app.core.errors import (
    DimensionMismatchError,
    DocumentNotFoundError,
    IndexingError,
    ModelMismatchError,
)
from app.core.logging import logger
from app.models.document import ProcessingStatus
from app.rag.retrieval.base import BaseVectorStore
from app.rag.retrieval.vector_store import vector_store
from app.repositories.chunk_repository import ChunkRepository, chunk_repository
from app.repositories.document_repository import DocumentRepository, document_repository
from app.repositories.embedding_repository import EmbeddingRepository, embedding_repository
from app.schemas.indexing import IndexDocumentResponse, IndexStatusResponse


class IndexingService:
    """
    Orchestrates the transfer of persisted structure-aware chunks and
    Phase 4 dense vector embeddings into ChromaDB with validation,
    idempotency guarantees, and status tracking.
    """

    def __init__(
        self,
        doc_repo: DocumentRepository = document_repository,
        chunk_repo: ChunkRepository = chunk_repository,
        embed_repo: EmbeddingRepository = embedding_repository,
        vector_store_instance: BaseVectorStore = vector_store,
    ):
        self.doc_repo = doc_repo
        self.chunk_repo = chunk_repo
        self.embed_repo = embed_repo
        self.vector_store = vector_store_instance

    async def index_document(
        self,
        document_id: str,
        force_reindex: bool = False,
    ) -> IndexDocumentResponse:
        """
        Validates and indexes existing persisted chunks and embeddings into ChromaDB.
        Does NOT silently regenerate embeddings or chunks.
        """
        # 1. Verify document exists
        doc = self.doc_repo.get(document_id)
        if not doc:
            raise DocumentNotFoundError(document_id)

        # 2. Retrieve persisted chunks
        chunks = self.chunk_repo.get_chunks_by_document(document_id)
        if not chunks:
            raise IndexingError(
                f"Document '{document_id}' has no chunks. Chunks must be generated before indexing.",
                status_code=400,
            )

        # 3. Retrieve persisted embeddings
        embeddings = self.embed_repo.get_embeddings_by_document(document_id, include_vectors=True)
        if not embeddings:
            raise IndexingError(
                f"Document '{document_id}' has no generated embeddings. "
                "Embeddings must be generated before indexing.",
                status_code=400,
            )

        # 4. Verify chunk IDs and count alignment
        if len(chunks) != len(embeddings):
            raise IndexingError(
                f"Chunk count ({len(chunks)}) does not match embedding count ({len(embeddings)}) "
                f"for document '{document_id}'.",
                status_code=400,
            )

        # Sort both by chunk_index to ensure deterministic alignment
        chunks_sorted = sorted(chunks, key=lambda c: c.chunk_index)
        embeddings_sorted = sorted(embeddings, key=lambda e: e.chunk_index)

        from app.rag.embeddings.embedding_service import embedding_service
        expected_model = embedding_service.provider.model_name
        expected_dimension = embedding_service.provider.dimension

        chunk_ids = []
        vectors = []
        texts = []
        metadatas = []

        for chunk, emb in zip(chunks_sorted, embeddings_sorted):
            if chunk.chunk_id != emb.chunk_id:
                raise IndexingError(
                    f"Chunk ID mismatch at index {chunk.chunk_index}: "
                    f"chunk='{chunk.chunk_id}' vs embedding='{emb.chunk_id}'.",
                    status_code=400,
                )
            if chunk.document_id != document_id or emb.document_id != document_id:
                raise IndexingError(
                    f"Document ID mismatch: expected '{document_id}', got chunk '{chunk.document_id}' / emb '{emb.document_id}'.",
                    status_code=400,
                )

            # 5. Verify model name and dimensions
            if emb.model_name != expected_model:
                raise ModelMismatchError(expected=expected_model, actual=emb.model_name)

            if emb.embedding_dimension != expected_dimension or len(emb.vector) != expected_dimension:
                raise DimensionMismatchError(expected=expected_dimension, actual=len(emb.vector))

            chunk_ids.append(chunk.chunk_id)
            vectors.append(emb.vector)
            texts.append(chunk.content)
            metadatas.append({
                "document_id": document_id,
                "chunk_id": chunk.chunk_id,
                "chunk_index": chunk.chunk_index,
                "source_filename": doc.original_filename,
                "file_type": doc.file_type.value if hasattr(doc.file_type, "value") else str(doc.file_type),
                "page_start": chunk.page_start if chunk.page_start is not None else -1,
                "page_end": chunk.page_end if chunk.page_end is not None else -1,
                "embedding_model": emb.model_name,
            })

        # 6. Insert into ChromaDB (executed in worker thread to prevent event-loop blocking)
        indexed_count = await asyncio.to_thread(
            self.vector_store.upsert_chunks,
            document_id=document_id,
            chunk_ids=chunk_ids,
            embeddings=vectors,
            documents=texts,
            metadatas=metadatas,
        )

        # 7. Update document status in SQLite
        self.doc_repo.update(
            document_id,
            processing_status=ProcessingStatus.INDEXED,
            vector_count=indexed_count,
        )

        collection_name = getattr(self.vector_store, "collection_name", "sift_document_embeddings")
        logger.info(
            f"Successfully indexed document '{document_id}' ({indexed_count} chunks) into collection '{collection_name}'"
        )

        return IndexDocumentResponse(
            document_id=document_id,
            status="indexed",
            indexed_chunks=indexed_count,
            total_chunks=len(chunks),
            embedding_model=expected_model,
            embedding_dimension=expected_dimension,
            collection_name=collection_name,
            message=f"Document successfully indexed into collection '{collection_name}'",
        )

    async def get_index_status(self, document_id: str) -> IndexStatusResponse:
        """Retrieves real vector indexing status for a document directly from ChromaDB."""
        doc = self.doc_repo.get(document_id)
        if not doc:
            raise DocumentNotFoundError(document_id)

        chunks = self.chunk_repo.get_chunks_by_document(document_id)
        total_chunks = len(chunks)

        indexed_chunks = await asyncio.to_thread(
            self.vector_store.get_document_chunk_count, document_id
        )

        model_name = self.embed_repo.get_model_name_for_document(document_id)
        collection_name = getattr(self.vector_store, "collection_name", "sift_document_embeddings")

        if indexed_chunks == 0:
            status = "not_indexed"
            is_indexed = False
        elif indexed_chunks == total_chunks and total_chunks > 0:
            status = "indexed"
            is_indexed = True
        else:
            status = "partial"
            is_indexed = False

        return IndexStatusResponse(
            document_id=document_id,
            status=status,
            is_indexed=is_indexed,
            indexed_chunks=indexed_chunks,
            total_chunks=total_chunks,
            embedding_model=model_name,
            embedding_dimension=384,
            collection_name=collection_name,
        )


# Default singleton instance
indexing_service = IndexingService()
