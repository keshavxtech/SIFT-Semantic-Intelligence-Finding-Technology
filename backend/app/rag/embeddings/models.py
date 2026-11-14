from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class ChunkEmbedding(BaseModel):
    """Domain model representing dense vector embedding for a specific chunk."""

    document_id: str = Field(..., description="Foreign key to source document")
    chunk_id: str = Field(..., description="Foreign key to exact source chunk")
    chunk_index: int = Field(..., description="0-based sequential chunk index")
    model_name: str = Field(..., description="Identifier of the model used to generate this vector")
    embedding_dimension: int = Field(default=384, description="Vector length / dimensionality")
    vector: List[float] = Field(..., description="Normalized float32 numerical vector coordinates")
    created_at: Optional[datetime] = Field(default_factory=datetime.utcnow)


class EmbeddingGenerationResponse(BaseModel):
    """Pydantic schema returned after generating/persisting embeddings."""

    document_id: str
    model_name: str
    embedding_dimension: int
    chunks_processed: int
    embeddings_created: int
    status: str = "completed"


class EmbeddingSummaryResponse(BaseModel):
    """Pydantic schema returned when inspecting existing document embeddings."""

    document_id: str
    model_name: str
    embedding_dimension: int
    total_embeddings: int
    status: str
    normalized: bool = True
