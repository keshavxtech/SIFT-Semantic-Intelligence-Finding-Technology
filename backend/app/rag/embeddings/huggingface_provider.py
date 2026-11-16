import math
from typing import Any, Dict, List, Optional
import httpx

from app.core.config import settings
from app.core.errors import EmbeddingProviderNotConfiguredError, SIFTException
from app.core.logging import logger
from app.rag.embeddings.base import BaseEmbeddingProvider
from app.rag.embeddings.config import EmbeddingConfig, DEFAULT_EMBEDDING_CONFIG


class HuggingFaceEmbeddingProvider(BaseEmbeddingProvider):
    """
    Hosted embedding provider using Hugging Face Inference API / Serverless API.
    Uses 'sentence-transformers/all-MiniLM-L6-v2' to produce normalized 384-dimensional dense vectors
    without requiring local PyTorch or SentenceTransformers runtime.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        api_url: Optional[str] = None,
        config: Optional[EmbeddingConfig] = None,
        timeout_seconds: float = 30.0,
        client: Optional[httpx.Client] = None,
    ):
        cfg = config or DEFAULT_EMBEDDING_CONFIG
        if model_name:
            cfg = EmbeddingConfig(
                model_name=model_name,
                dimension=cfg.dimension,
                batch_size=cfg.batch_size,
                normalize_embeddings=cfg.normalize_embeddings,
            )
        super().__init__(config=cfg)

        self._api_key = api_key or settings.HUGGINGFACE_API_KEY
        self._api_url = api_url or settings.HUGGINGFACE_API_URL
        self._timeout = timeout_seconds
        self._client = client

    @property
    def provider_name(self) -> str:
        return "huggingface"

    def _normalize_vector(self, vec: List[float]) -> List[float]:
        """Performs L2 vector normalization to unit length."""
        norm = math.sqrt(sum(v * v for v in vec))
        if norm < 1e-12:
            return vec
        return [v / norm for v in vec]

    def _extract_vector(self, raw_item: Any) -> List[float]:
        """
        Extracts a 1D vector from the raw API response item.
        Handles both 1D pooled vectors and 2D token-level matrices (performing mean pooling).
        """
        if not isinstance(raw_item, list):
            raise SIFTException(f"Unexpected item type from Hugging Face API: {type(raw_item)}", status_code=502)

        if len(raw_item) == 0:
            return [0.0] * self.dimension

        # Check if first element is float/int (already pooled 1D vector)
        if isinstance(raw_item[0], (int, float)):
            if len(raw_item) != self.dimension:
                raise SIFTException(
                    f"Hugging Face embedding dimension mismatch: got {len(raw_item)}, expected {self.dimension}",
                    status_code=502,
                )
            vec = [float(x) for x in raw_item]
            return self._normalize_vector(vec) if self.config.normalize_embeddings else vec

        # If it's a 2D matrix (token-level embeddings: seq_len x dimension), perform mean pooling
        if isinstance(raw_item[0], list):
            token_count = len(raw_item)
            dim = len(raw_item[0])
            if dim != self.dimension:
                raise SIFTException(
                    f"Hugging Face token embedding dimension mismatch: got {dim}, expected {self.dimension}",
                    status_code=502,
                )
            pooled = [0.0] * dim
            for token_vec in raw_item:
                for d in range(dim):
                    pooled[d] += float(token_vec[d])
            pooled = [val / token_count for val in pooled]
            return self._normalize_vector(pooled) if self.config.normalize_embeddings else pooled

        raise SIFTException("Unable to parse Hugging Face embedding format", status_code=502)

    def _make_request(self, texts: List[str]) -> Any:
        """Sends inference request to Hugging Face API."""
        headers: Dict[str, str] = {
            "Content-Type": "application/json",
            "X-Wait-For-Model": "true",
        }
        if self._api_key and self._api_key.strip():
            headers["Authorization"] = f"Bearer {self._api_key.strip()}"

        payload = {
            "inputs": texts,
            "options": {"wait_for_model": True},
        }

        if self._client is not None:
            response = self._client.post(self._api_url, json=payload, headers=headers, timeout=self._timeout)
        else:
            with httpx.Client(timeout=self._timeout) as http_client:
                response = http_client.post(self._api_url, json=payload, headers=headers)

        if response.status_code == 401:
            raise SIFTException(
                "Hugging Face API authentication failed. Please verify your HUGGINGFACE_API_KEY.",
                status_code=401,
            )
        if response.status_code == 503:
            raise SIFTException(
                "Hugging Face embedding model is currently loading. Please retry in a few moments.",
                status_code=503,
            )
        if response.status_code != 200:
            raise SIFTException(
                f"Hugging Face API returned error {response.status_code}: {response.text[:200]}",
                status_code=response.status_code if response.status_code in (400, 422, 429) else 502,
            )

        return response.json()

    def embed_text(self, text: str) -> List[float]:
        """Encodes a single text string into a normalized dense vector."""
        if not text or not text.strip():
            return [0.0] * self.dimension

        results = self.embed_batch([text])
        return results[0]

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Encodes a batch of text strings into normalized dense vectors."""
        if not texts:
            return []

        # Identify empty strings and prepare request list
        cleaned_texts: List[str] = []
        empty_indices = set()

        for idx, t in enumerate(texts):
            clean = (t or "").strip()
            if not clean:
                empty_indices.add(idx)
                cleaned_texts.append("empty passage")
            else:
                cleaned_texts.append(clean)

        raw_response = self._make_request(cleaned_texts)

        # Response can be a single vector if single input, or a list of vectors
        output_vectors: List[List[float]] = []

        if isinstance(raw_response, list):
            # Check if it's a single 1D vector returned for single input
            if len(cleaned_texts) == 1 and len(raw_response) > 0 and isinstance(raw_response[0], (int, float)):
                output_vectors.append(self._extract_vector(raw_response))
            else:
                for item in raw_response:
                    output_vectors.append(self._extract_vector(item))
        else:
            raise SIFTException("Unexpected non-list response from Hugging Face Inference API", status_code=502)

        # Replace vectors for originally empty strings with zero vectors
        for empty_idx in empty_indices:
            if empty_idx < len(output_vectors):
                output_vectors[empty_idx] = [0.0] * self.dimension

        return output_vectors
