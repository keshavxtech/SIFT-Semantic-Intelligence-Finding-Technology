import math
from abc import ABC, abstractmethod
from typing import List, Optional
from app.rag.embeddings.config import EmbeddingConfig, DEFAULT_EMBEDDING_CONFIG


class BaseEmbeddingProvider(ABC):
    """Abstract interface for local dense neural vector embedding models."""

    def __init__(self, config: Optional[EmbeddingConfig] = None):
        self.config = config or DEFAULT_EMBEDDING_CONFIG

    @property
    def model_name(self) -> str:
        return self.config.model_name

    @property
    def dimension(self) -> int:
        return self.config.dimension

    @abstractmethod
    def embed_text(self, text: str) -> List[float]:
        """Generates embedding for a single text string."""
        pass

    @abstractmethod
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generates embeddings for a batch of text strings."""
        pass

    def validate_vector(self, vector: List[float]) -> bool:
        """
        Validates that a vector matches the expected dimension and
        contains finite numerical floating point values.
        """
        if len(vector) != self.dimension:
            return False
        for val in vector:
            if not isinstance(val, (int, float)) or not math.isfinite(val):
                return False
        return True


class MockEmbeddingProvider(BaseEmbeddingProvider):
    """
    Deterministic mock provider for unit tests.
    Generates valid, normalized 384-dimensional vectors without external model downloads.
    """

    def __init__(self, config: Optional[EmbeddingConfig] = None):
        super().__init__(config=config or DEFAULT_EMBEDDING_CONFIG)

    def _generate_vector(self, text: str) -> List[float]:
        if not text:
            return [0.0] * self.dimension

        # Generate deterministic floats based on character codes and hash
        seed = abs(hash(text))
        raw_vals = [
            math.sin(seed + i * 0.17) + math.cos(seed * 0.05 + i * 0.31)
            for i in range(self.dimension)
        ]

        if self.config.normalize_embeddings:
            norm = math.sqrt(sum(v * v for v in raw_vals)) or 1.0
            return [v / norm for v in raw_vals]

        return raw_vals

    def embed_text(self, text: str) -> List[float]:
        if not text or not text.strip():
            return [0.0] * self.dimension
        vec = self._generate_vector(text)
        if not self.validate_vector(vec):
            raise ValueError(f"Generated vector failed validation for text '{text[:20]}'")
        return vec

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        return [self.embed_text(t) for t in texts]
