from datetime import datetime
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


class DocumentType(str, Enum):
    PDF = "pdf"
    TXT = "txt"
    MD = "md"
    DOCX = "docx"


class ProcessingStatus(str, Enum):
    PROCESSING = "processing"
    INDEXED = "indexed"
    FAILED = "failed"


class DocumentTag(str, Enum):
    TECHNICAL = "Technical"
    FINANCIAL = "Financial"
    SECURITY = "Security"
    LEGAL = "Legal"
    RESEARCH = "Research"
    COMPLIANCE = "Compliance"


class Document(BaseModel):
    id: str
    original_filename: str
    stored_filename: str
    name: str
    file_type: DocumentType
    file_size: int
    size: str
    upload_timestamp: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    processing_status: ProcessingStatus = ProcessingStatus.PROCESSING
    extracted_text: str = ""
    page_count: int = 1
    error_message: Optional[str] = None
    hash: str
    tag: DocumentTag = DocumentTag.TECHNICAL
    author: Optional[str] = "Workspace User"
    description: str = ""
    raw_text_preview: Optional[str] = None
    chunks_count: int = 0
    vector_count: int = 0
    token_count: int = 0
    file_path: Optional[str] = None  # Internal reference used by storage layer; never exposed in public schemas

    @property
    def type(self) -> DocumentType:
        return self.file_type

    @property
    def status(self) -> ProcessingStatus:
        return self.processing_status
