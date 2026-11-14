from typing import Optional
from fastapi import APIRouter, File, Form, Query, UploadFile, status
from app.models.document import Document
from app.rag.chunking.config import ChunkConfig, DEFAULT_CHUNK_CONFIG
from app.rag.chunking.models import ChunkResponse, DocumentChunksResponse
from app.rag.embeddings.embedding_service import embedding_service
from app.rag.embeddings.models import EmbeddingGenerationResponse, EmbeddingSummaryResponse
from app.schemas.indexing import IndexDocumentResponse, IndexStatusResponse
from app.schemas.document import (
    DocumentListResponse,
    DocumentRenameRequest,
    DocumentResponse,
)
from app.services.chunking_service import chunking_service
from app.services.document_service import document_service
from app.services.indexing_service import indexing_service

router = APIRouter(prefix="/documents", tags=["Documents"])


def to_document_response(doc: Document, include_extracted_text: bool = False) -> DocumentResponse:
    return DocumentResponse(
        id=doc.id,
        original_filename=doc.original_filename,
        stored_filename=doc.stored_filename,
        name=doc.name,
        file_type=doc.file_type,
        type=doc.file_type,
        file_size=doc.file_size,
        size_bytes=doc.file_size,
        size=doc.size,
        upload_timestamp=doc.upload_timestamp,
        uploaded_at=doc.upload_timestamp,
        processing_status=doc.processing_status,
        status=doc.processing_status,
        page_count=doc.page_count,
        error_message=doc.error_message,
        hash=doc.hash,
        tag=doc.tag,
        author=doc.author,
        description=doc.description,
        raw_text_preview=doc.raw_text_preview,
        extracted_text=doc.extracted_text if include_extracted_text else None,
        chunks_count=doc.chunks_count,
        vector_count=doc.vector_count,
        token_count=doc.token_count,
    )


@router.post("/upload", response_model=DocumentResponse, status_code=status.HTTP_201_CREATED)
async def upload_document(
    file: UploadFile = File(..., description="Document file to upload (.pdf, .txt)"),
    tag: str = Form("Technical", description="Knowledge classification tag"),
    description: str = Form("", description="Optional document summary description"),
) -> DocumentResponse:
    doc = await document_service.upload_document(file=file, tag=tag, description=description)
    return to_document_response(doc, include_extracted_text=False)


@router.get("", response_model=DocumentListResponse)
async def list_documents(
    tag: Optional[str] = None,
    status: Optional[str] = None,
    search: Optional[str] = None,
) -> DocumentListResponse:
    docs = document_service.list_documents(tag=tag, status=status, search=search)
    return DocumentListResponse(
        total=len(docs),
        documents=[to_document_response(d) for d in docs],
    )


@router.get("/{document_id}", response_model=DocumentResponse)
async def get_document(document_id: str) -> DocumentResponse:
    doc = document_service.get_document(document_id)
    return to_document_response(doc, include_extracted_text=True)


@router.patch("/{document_id}/rename", response_model=DocumentResponse)
async def rename_document(document_id: str, request: DocumentRenameRequest) -> DocumentResponse:
    doc = document_service.rename_document(document_id, request.name)
    return to_document_response(doc)


@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_document(document_id: str) -> None:
    document_service.delete_document(document_id)


@router.post("/{document_id}/chunks", response_model=DocumentChunksResponse)
async def generate_document_chunks(
    document_id: str,
    chunk_size: Optional[int] = Query(None, description="Override target chunk size in characters"),
    chunk_overlap: Optional[int] = Query(None, description="Override chunk overlap in characters"),
    minimum_chunk_size: Optional[int] = Query(None, description="Override minimum chunk size in characters"),
) -> DocumentChunksResponse:
    """
    Generates and persists deterministic, structure-aware chunks for the document.
    Re-running this endpoint idempotently regenerates and updates chunks.
    """
    custom_kwargs = {}
    if chunk_size is not None:
        custom_kwargs["chunk_size"] = chunk_size
    if chunk_overlap is not None:
        custom_kwargs["chunk_overlap"] = chunk_overlap
    if minimum_chunk_size is not None:
        custom_kwargs["minimum_chunk_size"] = minimum_chunk_size

    config = ChunkConfig(**custom_kwargs) if custom_kwargs else DEFAULT_CHUNK_CONFIG
    chunks = chunking_service.chunk_document(document_id=document_id, config=config, persist=True)

    return DocumentChunksResponse(
        document_id=document_id,
        total_chunks=len(chunks),
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
        chunks=[
            ChunkResponse(
                chunk_id=c.chunk_id,
                document_id=c.document_id,
                chunk_index=c.chunk_index,
                content=c.content,
                character_count=c.character_count,
                token_count=c.token_count,
                page_start=c.page_start,
                page_end=c.page_end,
                metadata=c.metadata,
            )
            for c in chunks
        ],
    )


@router.get("/{document_id}/chunks", response_model=DocumentChunksResponse)
async def get_document_chunks(document_id: str) -> DocumentChunksResponse:
    """
    Retrieves the ordered chunks for a document.
    Automatically generates and persists chunks if not yet generated.
    """
    chunks = chunking_service.get_document_chunks(document_id=document_id, auto_generate=True)
    return DocumentChunksResponse(
        document_id=document_id,
        total_chunks=len(chunks),
        chunk_size=DEFAULT_CHUNK_CONFIG.chunk_size,
        chunk_overlap=DEFAULT_CHUNK_CONFIG.chunk_overlap,
        chunks=[
            ChunkResponse(
                chunk_id=c.chunk_id,
                document_id=c.document_id,
                chunk_index=c.chunk_index,
                content=c.content,
                character_count=c.character_count,
                token_count=c.token_count,
                page_start=c.page_start,
                page_end=c.page_end,
                metadata=c.metadata,
            )
            for c in chunks
        ],
    )


@router.post("/{document_id}/embeddings", response_model=EmbeddingGenerationResponse)
async def generate_document_embeddings(
    document_id: str,
    force_recompute: bool = Query(False, description="Force re-generation even if matching model embeddings exist"),
) -> EmbeddingGenerationResponse:
    """
    Generates and persists 384-dimensional dense vector embeddings for all chunks of the document.
    Executes in a background thread to prevent blocking FastAPI's async event loop.
    """
    return await embedding_service.generate_document_embeddings(
        document_id=document_id,
        force_recompute=force_recompute,
    )


@router.get("/{document_id}/embeddings", response_model=EmbeddingSummaryResponse)
async def get_document_embeddings_summary(document_id: str) -> EmbeddingSummaryResponse:
    """
    Retrieves summary metadata and status for document embeddings.
    Does not transmit large raw vector arrays over the wire.
    """
    return embedding_service.get_document_embeddings_summary(document_id=document_id)


@router.post("/{document_id}/index", response_model=IndexDocumentResponse)
async def index_document(
    document_id: str,
    force_reindex: bool = Query(False, description="Force re-indexing even if already indexed"),
) -> IndexDocumentResponse:
    """
    Transfers persisted Phase 4 embeddings and structure-aware chunk metadata into ChromaDB.
    Guarantees idempotency and safe re-indexing with cosine distance metric.
    """
    return await indexing_service.index_document(document_id=document_id, force_reindex=force_reindex)


@router.get("/{document_id}/index-status", response_model=IndexStatusResponse)
async def get_document_index_status(document_id: str) -> IndexStatusResponse:
    """
    Retrieves the current vector indexing status and chunk counts for a document from ChromaDB.
    """
    return await indexing_service.get_index_status(document_id=document_id)
