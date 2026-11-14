from pathlib import Path
from typing import Dict, List, Optional
from app.core.errors import DocumentNotFoundError, NoExtractableTextError
from app.core.logging import logger
from app.models.document import Document, DocumentType
from app.rag.chunking.config import ChunkConfig, DEFAULT_CHUNK_CONFIG
from app.rag.chunking.models import Chunk
from app.rag.chunking.semantic_chunker import StructureAwareChunker
from app.rag.ingestion.parser import PDFParser, ParsedPage
from app.repositories.chunk_repository import ChunkRepository, chunk_repository
from app.repositories.document_repository import DocumentRepository, document_repository


class ChunkingService:
    """Service orchestrating document segmentation, metadata preservation, and chunk persistence."""

    def __init__(
        self,
        doc_repo: DocumentRepository = document_repository,
        chunk_repo: ChunkRepository = chunk_repository,
    ):
        self.doc_repo = doc_repo
        self.chunk_repo = chunk_repo

    def _resolve_pages_for_document(self, doc: Document) -> Optional[List[ParsedPage]]:
        """
        Extracts verified page structures for PDF documents.
        Returns None for TXT or documents where physical pages are not applicable.
        """
        if doc.file_type != DocumentType.PDF:
            return None

        # If physical file is available on disk, extract exact page-by-page text
        if doc.file_path and Path(doc.file_path).exists():
            try:
                with open(doc.file_path, "rb") as f:
                    file_bytes = f.read()
                extraction = PDFParser.parse(file_bytes, filename=doc.name)
                return extraction.pages
            except Exception as e:
                logger.warning(f"Could not re-parse PDF file at {doc.file_path} for pages: {e}")

        # Fallback for single page or simulated documents without physical file
        if doc.page_count == 1:
            return [ParsedPage(page_number=1, text=doc.extracted_text)]

        return None

    def chunk_document(
        self,
        document_id: str,
        config: Optional[ChunkConfig] = None,
        persist: bool = True,
    ) -> List[Chunk]:
        """
        Generates deterministic, structure-aware chunks for a document.
        Optionally persists the generated chunks to SQLite (idempotent).
        """
        doc = self.doc_repo.get(document_id)
        if not doc:
            raise DocumentNotFoundError(document_id)

        if not doc.extracted_text or not doc.extracted_text.strip():
            logger.warning(f"Document {document_id} ('{doc.name}') contains no extractable text for chunking.")
            raise NoExtractableTextError(doc.name)

        effective_config = config or DEFAULT_CHUNK_CONFIG
        chunker = StructureAwareChunker(config=effective_config)

        # Build provenance metadata (strictly omitting internal filesystem paths)
        metadata = {
            "document_name": doc.name,
            "original_filename": doc.original_filename,
            "file_type": doc.file_type.value,
            "tag": doc.tag.value if hasattr(doc.tag, "value") else str(doc.tag),
            "upload_timestamp": doc.upload_timestamp,
        }

        # Resolve page structure (PDF has 1-based page numbers; TXT has None)
        pages = self._resolve_pages_for_document(doc)

        chunks = chunker.chunk_document(
            document_id=doc.id,
            text=doc.extracted_text,
            pages=pages,
            metadata=metadata,
        )

        if persist and chunks:
            self.chunk_repo.save_chunks(document_id=doc.id, chunks=chunks)
            # Update document repository in-memory / disk state
            self.doc_repo.update(document_id=doc.id, chunks_count=len(chunks))

        logger.info(
            f"Chunked document '{doc.name}' ({doc.id}) into {len(chunks)} chunks "
            f"(size={effective_config.chunk_size}, overlap={effective_config.chunk_overlap})"
        )
        return chunks

    def get_document_chunks(
        self,
        document_id: str,
        auto_generate: bool = True,
    ) -> List[Chunk]:
        """
        Retrieves existing chunks for a document, or generates them on demand if not yet chunked.
        """
        doc = self.doc_repo.get(document_id)
        if not doc:
            raise DocumentNotFoundError(document_id)

        existing = self.chunk_repo.get_chunks_by_document(document_id)
        if existing:
            return existing

        if auto_generate:
            return self.chunk_document(document_id=document_id, persist=True)

        return []


# Default singleton instance
chunking_service = ChunkingService()
