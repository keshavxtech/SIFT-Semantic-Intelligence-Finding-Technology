from pathlib import Path
from typing import Optional
from pydantic import BaseModel, Field, model_validator


class EmbeddingConfig(BaseModel):
    """Configuration for local sentence-transformers embedding generation."""

    model_name: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="HuggingFace model identifier for local dense embeddings.",
    )
    dimension: int = Field(
        default=384,
        description="Expected embedding vector dimension for all-MiniLM-L6-v2.",
    )
    batch_size: int = Field(
        default=32,
        description="Number of chunks to process per inference batch.",
    )
    normalize_embeddings: bool = Field(
        default=True,
        description="Whether to L2-normalize vectors to unit length.",
    )
    device: str = Field(
        default="cpu",
        description="Compute device for PyTorch inference (CPU by default).",
    )
    model_cache_dir: Optional[str] = Field(
        default="data/models",
        description="Local directory for caching downloaded model weights.",
    )

    @model_validator(mode="after")
    def validate_config(self) -> "EmbeddingConfig":
        if self.dimension <= 0:
            raise ValueError(f"dimension must be positive, got {self.dimension}")
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")
        return self


# Default configuration singleton
DEFAULT_EMBEDDING_CONFIG = EmbeddingConfig()
