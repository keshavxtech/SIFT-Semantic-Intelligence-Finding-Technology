from pydantic import BaseModel, Field, model_validator


class ChunkConfig(BaseModel):
    """
    Centralized configuration for deterministic structure-aware chunking.
    All dimensions are measured in characters.
    """
    chunk_size: int = Field(default=1000, description="Target maximum character count per chunk.")
    chunk_overlap: int = Field(default=150, description="Maximum character overlap between consecutive chunks.")
    minimum_chunk_size: int = Field(default=100, description="Minimum character threshold for standalone chunks.")

    @model_validator(mode="after")
    def validate_chunk_bounds(self) -> "ChunkConfig":
        if self.chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {self.chunk_size}")

        if self.chunk_overlap < 0:
            raise ValueError(f"chunk_overlap cannot be negative, got {self.chunk_overlap}")

        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                f"chunk_overlap ({self.chunk_overlap}) must be smaller than chunk_size ({self.chunk_size})"
            )

        if self.minimum_chunk_size <= 0:
            raise ValueError(f"minimum_chunk_size must be positive, got {self.minimum_chunk_size}")

        if self.minimum_chunk_size > self.chunk_size:
            raise ValueError(
                f"minimum_chunk_size ({self.minimum_chunk_size}) must not exceed chunk_size ({self.chunk_size})"
            )

        return self


# Default configuration singleton
DEFAULT_CHUNK_CONFIG = ChunkConfig()
