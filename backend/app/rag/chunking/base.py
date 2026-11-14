from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from app.rag.chunking.config import ChunkConfig, DEFAULT_CHUNK_CONFIG
from app.rag.chunking.models import Chunk
from app.rag.ingestion.parser import ParsedPage


class BaseChunker(ABC):
    """Abstract base class for document chunking engines."""

    def __init__(self, config: Optional[ChunkConfig] = None):
        self.config = config or DEFAULT_CHUNK_CONFIG

    @abstractmethod
    def chunk_document(
        self,
        document_id: str,
        text: str,
        pages: Optional[List[ParsedPage]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Chunk]:
        """
        Chunks input text or page structures into an ordered list of Chunks.
        Must be deterministic and preserve page provenance where available.
        """
        pass
