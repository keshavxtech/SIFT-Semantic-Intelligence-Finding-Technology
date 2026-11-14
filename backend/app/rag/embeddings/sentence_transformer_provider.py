import threading
from pathlib import Path
from typing import Any, List, Optional
from app.core.logging import logger
from app.rag.embeddings.base import BaseEmbeddingProvider
from app.rag.embeddings.config import EmbeddingConfig, DEFAULT_EMBEDDING_CONFIG


class SentenceTransformerProvider(BaseEmbeddingProvider):
    """
    Local embedding provider using SentenceTransformers.
    Supports lazy loading, CPU execution, local caching, and batch inference.
    """

    def __init__(self, config: Optional[EmbeddingConfig] = None):
        super().__init__(config=config or DEFAULT_EMBEDDING_CONFIG)
        self._model: Optional[Any] = None
        self._lock = threading.Lock()

    def _get_model(self) -> Any:
        """Lazily initializes and reuses the SentenceTransformer model instance."""
        if self._model is None:
            with self._lock:
                if self._model is None:
                    # Import inside method to avoid heavy import-time overhead
                    from sentence_transformers import SentenceTransformer

                    cache_dir = None
                    if self.config.model_cache_dir:
                        cache_path = Path(self.config.model_cache_dir)
                        cache_path.mkdir(parents=True, exist_ok=True)
                        cache_dir = str(cache_path.resolve())

                    logger.info(
                        f"Loading embedding model '{self.config.model_name}' "
                        f"on device '{self.config.device}' (cache: {cache_dir or 'default'})..."
                    )
                    try:
                        self._model = SentenceTransformer(
                            model_name_or_path=self.config.model_name,
                            device=self.config.device,
                            cache_folder=cache_dir,
                            local_files_only=True,
                        )
                    except Exception:
                        self._model = SentenceTransformer(
                            model_name_or_path=self.config.model_name,
                            device=self.config.device,
                            cache_folder=cache_dir,
                        )
                    logger.info(f"Successfully loaded '{self.config.model_name}' (dim={self.dimension})")

        return self._model

    def embed_text(self, text: str) -> List[float]:
        """Encodes a single text string into a normalized dense vector."""
        if not text or not text.strip():
            return [0.0] * self.dimension

        results = self.embed_batch([text])
        return results[0] if results else [0.0] * self.dimension

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Encodes a batch of text strings into normalized dense vectors."""
        if not texts:
            return []

        model = self._get_model()

        # Execute batch encoding with sentence-transformers
        raw_embeddings = model.encode(
            texts,
            batch_size=self.config.batch_size,
            show_progress_bar=False,
            normalize_embeddings=self.config.normalize_embeddings,
            convert_to_numpy=True,
        )

        vectors: List[List[float]] = []
        for idx, vec in enumerate(raw_embeddings):
            vec_list = [float(v) for v in vec]
            if not self.validate_vector(vec_list):
                raise ValueError(
                    f"Generated vector at index {idx} failed dimension or finite float validation "
                    f"(expected dimension {self.dimension}, got {len(vec_list)})"
                )
            vectors.append(vec_list)

        return vectors
