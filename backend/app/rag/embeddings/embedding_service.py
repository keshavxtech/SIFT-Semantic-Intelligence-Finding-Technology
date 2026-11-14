import asyncio
from typing import List, Optional
from app.core.errors import DocumentNotFoundError, NoExtractableTextError
from app.core.logging import logger
from app.rag.chunking.models import Chunk
from app.rag.embeddings.base import BaseEmbeddingProvider
from app.rag.embeddings.config import EmbeddingConfig, DEFAULT_EMBEDDING_CONFIG
from app.rag.embeddings.models import (
    ChunkEmbedding,
    EmbeddingGenerationResponse,
    EmbeddingSummaryResponse,
)
from app.rag.embeddings.sentence_transformer_provider import SentenceTransformerProvider
from app.repositories.document_repository import DocumentRepository, document_repository
from app.repositories.embedding_repository import EmbeddingRepository, embedding_repository
from app.services.chunking_service import ChunkingService, chunking_service


class EmbeddingService:
    """Service orchestrating batch vector generation, provenance tracking, and SQLite persistence."""

    def __init__(
        self,
        provider: Optional[BaseEmbeddingProvider] = None,
        doc_repo: DocumentRepository = document_repository,
        embed_repo: EmbeddingRepository = embedding_repository,
        chunk_svc: ChunkingService = chunking_service,
    ):
        self.provider = provider or SentenceTransformerProvider(config=DEFAULT_EMBEDDING_CONFIG)
        self.doc_repo = doc_repo
        self.embed_repo = embed_repo
        self.chunk_svc = chunk_svc

    async def generate_document_embeddings(
        self,
        document_id: str,
        force_recompute: bool = False,
    ) -> EmbeddingGenerationResponse:
        """
        Generates and persists dense vector embeddings for all chunks in a document.
        Executes CPU-bound batch inference in a background thread to avoid blocking the async event loop.
        """
        doc = self.doc_repo.get(document_id)
        if not doc:
            raise DocumentNotFoundError(document_id)

        # Retrieve chunks (auto-generates if not yet chunked)
        chunks: List[Chunk] = self.chunk_svc.get_document_chunks(document_id, auto_generate=True)
        if not chunks:
            logger.warning(f"Document '{doc.name}' ({document_id}) has no chunks available to embed.")
            raise NoExtractableTextError(doc.name)

        # Check existing embeddings for idempotency & model mismatch
        existing_model = self.embed_repo.get_model_name_for_document(document_id)
        existing_count = self.embed_repo.get_count(document_id)

        if (
            not force_recompute
            and existing_model == self.provider.model_name
            and existing_count == len(chunks)
        ):
            logger.info(
                f"Document '{document_id}' already has {existing_count} embeddings "
                f"using model '{self.provider.model_name}'. Reusing existing vectors."
            )
            return EmbeddingGenerationResponse(
                document_id=document_id,
                model_name=self.provider.model_name,
                embedding_dimension=self.provider.dimension,
                chunks_processed=len(chunks),
                embeddings_created=existing_count,
                status="completed",
            )

        # Prepare text list for batch encoding
        texts = [c.content for c in chunks]

        # Execute batch embedding in a worker thread so the async event loop is never blocked
        vectors = await asyncio.to_thread(self.provider.embed_batch, texts)

        if len(vectors) != len(chunks):
            raise ValueError(
                f"Embedding count mismatch: expected {len(chunks)} vectors, got {len(vectors)}"
            )

        # Assemble ChunkEmbedding records maintaining strict 1:1 order
        chunk_embeddings: List[ChunkEmbedding] = []
        for idx, (chunk, vec) in enumerate(zip(chunks, vectors)):
            chunk_embeddings.append(
                ChunkEmbedding(
                    document_id=document_id,
                    chunk_id=chunk.chunk_id,
                    chunk_index=chunk.chunk_index,
                    model_name=self.provider.model_name,
                    embedding_dimension=self.provider.dimension,
                    vector=vec,
                )
            )

        # Persist vectors atomically to SQLite
        self.embed_repo.save_embeddings(document_id=document_id, embeddings=chunk_embeddings)
        self.doc_repo.update(document_id=document_id, vector_count=len(chunk_embeddings))

        logger.info(
            f"Successfully generated {len(chunk_embeddings)} embeddings for document '{doc.name}' "
            f"({document_id}) using '{self.provider.model_name}'"
        )

        return EmbeddingGenerationResponse(
            document_id=document_id,
            model_name=self.provider.model_name,
            embedding_dimension=self.provider.dimension,
            chunks_processed=len(chunks),
            embeddings_created=len(chunk_embeddings),
            status="completed",
        )

    def get_document_embeddings_summary(self, document_id: str) -> EmbeddingSummaryResponse:
        """Retrieves summary metadata for document embeddings without returning raw vectors."""
        doc = self.doc_repo.get(document_id)
        if not doc:
            raise DocumentNotFoundError(document_id)

        count = self.embed_repo.get_count(document_id)
        model_name = self.embed_repo.get_model_name_for_document(document_id) or self.provider.model_name

        return EmbeddingSummaryResponse(
            document_id=document_id,
            model_name=model_name,
            embedding_dimension=self.provider.dimension,
            total_embeddings=count,
            status="completed" if count > 0 else "pending",
            normalized=self.provider.config.normalize_embeddings,
        )


# Default singleton instance
embedding_service = EmbeddingService()
