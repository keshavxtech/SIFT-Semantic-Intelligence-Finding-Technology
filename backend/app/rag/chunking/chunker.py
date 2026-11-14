from typing import List, Dict, Any


class TextChunker:
    """Sliding-window chunker with token boundary preservation."""

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 64):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_text(self, text: str, document_id: str, document_name: str) -> List[Dict[str, Any]]:
        """Splits full text into chunks with token approximations and metadata."""
        if not text.strip():
            return []

        words = text.split()
        chunks: List[Dict[str, Any]] = []
        # Approximate 1 token ~ 0.75 words => words_per_chunk ~ chunk_size * 0.75
        words_per_chunk = max(30, int(self.chunk_size * 0.75))
        words_overlap = max(5, int(self.chunk_overlap * 0.75))
        step = max(1, words_per_chunk - words_overlap)

        idx = 0
        for start in range(0, len(words), step):
            chunk_words = words[start : start + words_per_chunk]
            if not chunk_words:
                break
            chunk_str = " ".join(chunk_words)
            chunk_id = f"chk_{document_id}_{idx:03d}"
            # Approximate tokens
            token_count = max(1, int(len(chunk_words) / 0.75))

            chunks.append({
                "chunk_id": chunk_id,
                "document_id": document_id,
                "document_name": document_name,
                "passage": chunk_str,
                "token_count": token_count,
                "index": idx,
            })
            idx += 1
            if start + words_per_chunk >= len(words):
                break

        return chunks
