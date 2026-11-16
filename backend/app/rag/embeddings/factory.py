from typing import Optional
from app.core.config import settings
from app.rag.embeddings.base import BaseEmbeddingProvider, MockEmbeddingProvider
from app.rag.embeddings.config import DEFAULT_EMBEDDING_CONFIG


def get_embedding_provider(provider_type: Optional[str] = None) -> BaseEmbeddingProvider:
    """
    Factory resolving the active dense vector embedding provider based on configuration.
    Supports local SentenceTransformers, hosted Hugging Face Inference API, and deterministic mocks.
    """
    name = (provider_type or settings.EMBEDDING_PROVIDER).lower().strip()

    if name in ("huggingface", "hf", "hosted"):
        from app.rag.embeddings.huggingface_provider import HuggingFaceEmbeddingProvider
        return HuggingFaceEmbeddingProvider()

    elif name == "mock":
        return MockEmbeddingProvider(config=DEFAULT_EMBEDDING_CONFIG)

    else:
        # Default local sentence-transformers
        from app.rag.embeddings.sentence_transformer_provider import SentenceTransformerProvider
        return SentenceTransformerProvider(config=DEFAULT_EMBEDDING_CONFIG)
