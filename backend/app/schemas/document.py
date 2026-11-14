from typing import List, Optional
from pydantic import BaseModel, Field
from app.models.document import DocumentType, ProcessingStatus, DocumentTag


class DocumentResponse(BaseModel):
    id: str
    original_filename: str
    stored_filename: str
    name: str
    file_type: DocumentType
    type: DocumentType  # Frontend & existing API compatibility alias
    file_size: int
    size_bytes: int  # Compatibility alias
    size: str
    upload_timestamp: str
    uploaded_at: str  # Compatibility alias
    processing_status: ProcessingStatus
    status: ProcessingStatus  # Compatibility alias
    page_count: int
    error_message: Optional[str] = None
    hash: str
    tag: DocumentTag
    author: Optional[str] = "Workspace User"
    description: str = ""
    raw_text_preview: Optional[str] = None
    extracted_text: Optional[str] = None
    chunks_count: int = 0
    vector_count: int = 0
    token_count: int = 0


class DocumentDetailResponse(DocumentResponse):
    """Full detail response including full extracted text."""
    extracted_text: str


class DocumentListResponse(BaseModel):
    total: int
    documents: List[DocumentResponse]


class DocumentRenameRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
