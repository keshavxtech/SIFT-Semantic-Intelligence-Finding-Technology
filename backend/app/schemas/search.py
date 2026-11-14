from typing import List, Optional
from pydantic import BaseModel, Field
from app.models.document import DocumentTag, DocumentType


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Natural-language search query")
    document_ids: Optional[List[str]] = Field(default=None, description="Scope search to specific documents")
    top_k: Optional[int] = Field(default=None, ge=1, le=20, description="Number of results to return (1-20)")
    limit: Optional[int] = Field(default=None, ge=1, le=50, description="Legacy limit parameter")
    threshold: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Minimum cosine similarity threshold")
    mode: str = Field(default="dense", description="Retrieval mode: dense, hybrid, or bm25")

    @property
    def effective_top_k(self) -> int:
        if self.top_k is not None:
            return max(1, min(20, self.top_k))
        if self.limit is not None:
            return max(1, min(20, self.limit))
        return 5


class SearchResultItem(BaseModel):
    # Core Phase 5 Fields
    document_id: str = Field(..., description="Parent document identifier")
    chunk_id: str = Field(..., description="Unique persistent chunk identifier")
    chunk_index: int = Field(..., description="0-based sequence index within document")
    content: str = Field(..., description="Text content of the retrieved chunk passage")
    source_filename: str = Field(..., description="Original filename of the source document")
    page_start: Optional[int] = Field(default=None, description="1-based start page (null for TXT)")
    page_end: Optional[int] = Field(default=None, description="1-based end page (null for TXT)")
    distance: float = Field(..., description="Cosine distance from query vector (lower is better, range [0, 2])")
    similarity_score: float = Field(..., description="Cosine similarity score (1 - distance, higher is better)")

    # Frontend Compatibility & UI Alignment Fields
    id: str = Field(..., description="Result entry identifier")
    docId: str = Field(..., description="Alias for document_id")
    docName: str = Field(..., description="Alias for source_filename")
    docType: DocumentType = Field(default=DocumentType.TXT, description="Document type")
    pageNumber: int = Field(default=1, description="Representative page number")
    chunkId: str = Field(..., description="Alias for chunk_id")
    passage: str = Field(..., description="Alias for content")
    score: float = Field(..., description="Relevance score (alias for similarity_score)")
    cosineDistance: float = Field(..., description="Alias for distance")
    tokenCount: int = Field(default=0, description="Estimated token count")
    matchHighlights: List[str] = Field(default_factory=list, description="Extracted matched terms")
    sectionTitle: str = Field(default="", description="Section label or provenance header")
    tag: DocumentTag = Field(default=DocumentTag.TECHNICAL, description="Document classification tag")


class SearchResponse(BaseModel):
    query: str
    total_matches: int
    execution_time_ms: float
    mode: str
    metric: str = Field(default="cosine_distance (lower is better)")
    results: List[SearchResultItem]
