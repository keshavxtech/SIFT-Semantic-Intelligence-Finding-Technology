from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


@dataclass
class TextSegment:
    """Internal atomic text unit with page-level provenance."""
    text: str
    page_number: Optional[int] = None


class Chunk(BaseModel):
    """Domain model for a deterministic document chunk."""
    chunk_id: str = Field(..., description="Unique stable chunk identifier, e.g. chk_{doc_id}_{index:04d}")
    document_id: str = Field(..., description="Foreign key to source document identifier")
    chunk_index: int = Field(..., description="0-based sequential order within the document")
    content: str = Field(..., description="Text content of the chunk")
    character_count: int = Field(..., description="Length of content in characters")
    token_count: int = Field(default=0, description="Approximate token count for the chunk")
    page_start: Optional[int] = Field(default=None, description="Starting 1-based PDF page number (null for TXT)")
    page_end: Optional[int] = Field(default=None, description="Ending 1-based PDF page number (null for TXT)")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Source provenance metadata (filename, type, etc.)")
    created_at: Optional[datetime] = Field(default_factory=datetime.utcnow)


class ChunkResponse(BaseModel):
    """Pydantic schema for API chunk serialization."""
    chunk_id: str
    document_id: str
    chunk_index: int
    content: str
    character_count: int
    token_count: int
    page_start: Optional[int] = None
    page_end: Optional[int] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class DocumentChunksResponse(BaseModel):
    """Pydantic schema for document chunking endpoint response."""
    document_id: str
    total_chunks: int
    chunk_size: int
    chunk_overlap: int
    chunks: List[ChunkResponse]
