import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Optional
from fastapi import UploadFile

from app.core.config import settings
from app.core.errors import (
    CorruptedDocumentError,
    DocumentNotFoundError,
    EmptyFileError,
    FileSizeLimitExceededError,
    NoExtractableTextError,
    StorageError,
    UnsupportedFileTypeError,
)
from app.core.logging import logger
from app.models.document import Document, DocumentTag, DocumentType, ProcessingStatus
from app.rag.ingestion.parser import DocumentParser
from app.repositories.document_repository import DocumentRepository, document_repository
from app.utils.file_utils import (
    compute_sha256,
    format_bytes_to_human,
    is_valid_pdf_signature,
    sanitize_filename,
    validate_file_extension,
    validate_mime_type,
)


class DocumentService:
    def __init__(self, repo: DocumentRepository = document_repository):
        self.repo = repo

    async def upload_document(
        self,
        file: UploadFile,
        tag: str = "Technical",
        description: str = "",
    ) -> Document:
        raw_filename = file.filename or "untitled.txt"
        sanitized_filename = sanitize_filename(raw_filename)

        # 1. Validate file extension
        valid_ext, ext = validate_file_extension(sanitized_filename)
        if not valid_ext:
            logger.warning(f"Rejected unsupported file extension: '{ext}' for file '{raw_filename}'")
            raise UnsupportedFileTypeError(ext)

        # 2. Validate MIME type
        if file.content_type and not validate_mime_type(file.content_type, ext):
            logger.warning(
                f"MIME type mismatch for file '{sanitized_filename}': {file.content_type} with extension {ext}"
            )
            # Permissive if extension is genuinely valid, but log warning

        # 3. Read uploaded bytes
        content = await file.read()
        size_bytes = len(content)

        # 4. Reject empty files
        if size_bytes == 0:
            logger.warning(f"Rejected empty upload for file '{sanitized_filename}'")
            raise EmptyFileError(sanitized_filename)

        # 5. Reject oversized files
        size_mb = size_bytes / (1024 * 1024)
        if size_mb > settings.MAX_UPLOAD_SIZE_MB:
            logger.warning(
                f"File size {size_mb:.2f} MB exceeds limit {settings.MAX_UPLOAD_SIZE_MB} MB for '{sanitized_filename}'"
            )
            raise FileSizeLimitExceededError(size_mb, settings.MAX_UPLOAD_SIZE_MB)

        # 6. Basic format integrity check
        if ext == "pdf" and not is_valid_pdf_signature(content):
            logger.warning(f"File '{sanitized_filename}' failed PDF magic header verification")
            raise CorruptedDocumentError(sanitized_filename, reason="File does not have a valid PDF header signature.")

        # 7. Generate identifiers and storage paths
        doc_id = f"doc-{uuid.uuid4().hex[:12]}"
        stored_filename = f"{doc_id}_{sanitized_filename}"

        # Prevent path traversal and enforce storage boundary
        storage_dir = settings.storage_path.resolve()
        target_path = (storage_dir / stored_filename).resolve()

        if not str(target_path).startswith(str(storage_dir)):
            logger.error(f"Path traversal detected during upload: {sanitized_filename}")
            raise StorageError("Invalid storage path destination.")

        # Ensure no accidental overwriting
        counter = 1
        while target_path.exists():
            stem = Path(stored_filename).stem
            suffix = Path(stored_filename).suffix
            stored_filename = f"{stem}_{counter}{suffix}"
            target_path = (storage_dir / stored_filename).resolve()
            counter += 1

        # 8. Store physical file
        try:
            with open(target_path, "wb") as f:
                f.write(content)
        except Exception as e:
            logger.error(f"Failed to write file to disk: {target_path} ({e})")
            raise StorageError(f"Failed to save file to disk: {str(e)}")

        # 9. Parse and extract text
        try:
            extraction = DocumentParser.extract_content(
                file_bytes=content,
                filename=sanitized_filename,
                file_type=ext,
            )
        except Exception as e:
            # Clean up the stored file on parsing failure to avoid orphan disk files
            if target_path.exists():
                try:
                    os.remove(target_path)
                except OSError:
                    pass
            logger.warning(f"Document parsing failed for '{sanitized_filename}': {e}")
            raise

        # 10. Format metadata and persist
        file_hash = compute_sha256(content)
        human_size = format_bytes_to_human(size_bytes)
        now_iso = datetime.utcnow().isoformat()

        try:
            domain_tag = DocumentTag(tag)
        except ValueError:
            domain_tag = DocumentTag.TECHNICAL

        doc = Document(
            id=doc_id,
            original_filename=raw_filename,
            stored_filename=stored_filename,
            name=sanitized_filename,
            file_type=DocumentType(ext),
            file_size=size_bytes,
            size=human_size,
            upload_timestamp=now_iso,
            processing_status=ProcessingStatus.INDEXED,
            extracted_text=extraction.full_text,
            page_count=extraction.page_count,
            error_message=None,
            hash=file_hash,
            tag=domain_tag,
            description=description or f"Uploaded {ext.upper()} document ({extraction.page_count} page{'s' if extraction.page_count != 1 else ''}).",
            raw_text_preview=extraction.preview_snippet,
            chunks_count=0,  # No chunking in this phase per scope boundaries
            vector_count=0,
            token_count=len(extraction.full_text.split()),
            file_path=str(target_path),
        )

        created_doc = self.repo.create(doc)
        logger.info(
            f"Successfully ingested document '{doc.name}' (id: {doc_id}, pages: {doc.page_count}, size: {doc.size})"
        )
        return created_doc

    def get_document(self, document_id: str) -> Document:
        doc = self.repo.get(document_id)
        if not doc:
            raise DocumentNotFoundError(document_id)
        return doc

    def list_documents(
        self,
        tag: Optional[str] = None,
        status: Optional[str] = None,
        search: Optional[str] = None,
    ) -> List[Document]:
        return self.repo.list_all(tag=tag, status=status, search=search)

    def rename_document(self, document_id: str, new_name: str) -> Document:
        doc = self.get_document(document_id)
        clean_name = sanitize_filename(new_name.strip())
        updated = self.repo.update(document_id, name=clean_name)
        logger.info(f"Renamed document {document_id} to '{clean_name}'")
        return updated or doc

    def delete_document(self, document_id: str) -> bool:
        doc = self.get_document(document_id)

        # 1. Safely remove physical file
        if doc.file_path and os.path.exists(doc.file_path):
            try:
                os.remove(doc.file_path)
                logger.info(f"Removed physical storage file: {doc.file_path}")
            except OSError as e:
                logger.warning(f"Could not remove physical file at {doc.file_path}: {e}")

        # 2. Remove associated chunks, embeddings, and vector store records
        from app.repositories.chunk_repository import chunk_repository
        from app.repositories.embedding_repository import embedding_repository
        from app.rag.retrieval.vector_store import vector_store

        chunk_repository.delete_chunks_by_document(document_id)
        embedding_repository.delete_embeddings_by_document(document_id)
        vector_store.delete_document(document_id)

        # 3. Remove database record
        deleted = self.repo.delete(document_id)
        logger.info(f"Deleted document record {document_id}")
        return deleted


# Default singleton instance
document_service = DocumentService()
