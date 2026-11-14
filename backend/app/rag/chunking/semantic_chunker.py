import re
from typing import Any, Dict, List, Optional
from app.rag.chunking.base import BaseChunker
from app.rag.chunking.config import ChunkConfig, DEFAULT_CHUNK_CONFIG
from app.rag.chunking.models import Chunk, TextSegment
from app.rag.ingestion.parser import ParsedPage


class StructureAwareChunker(BaseChunker):
    """
    Deterministic, structure-aware document chunker.
    
    Preserves natural document hierarchy:
    1. Paragraph boundaries (\n\n)
    2. Sentence boundaries (. ! ?)
    3. Clause / phrase boundaries (; , :)
    4. Word boundaries
    
    Preserves exact page provenance (page_start, page_end) when page structures are provided.
    Does not use LLMs or embeddings, guaranteeing deterministic, reproducible segmentation.
    """

    # Sentence boundary regex: punctuation followed by space and capital letter/digit/quote
    SENTENCE_SPLIT_REGEX = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])")
    # Clause boundary regex for unusually long sentences
    CLAUSE_SPLIT_REGEX = re.compile(r"(?<=[;:,])\s+")

    def __init__(self, config: Optional[ChunkConfig] = None):
        super().__init__(config=config or DEFAULT_CHUNK_CONFIG)

    def _split_oversized_text(self, text: str, page_number: Optional[int]) -> List[TextSegment]:
        """Recursively breaks oversized text into sentences, clauses, or words."""
        max_size = self.config.chunk_size

        if len(text) <= max_size:
            return [TextSegment(text=text, page_number=page_number)]

        # Try splitting on sentence boundaries
        sentences = self.SENTENCE_SPLIT_REGEX.split(text)
        if len(sentences) > 1:
            segments: List[TextSegment] = []
            for s in sentences:
                s_stripped = s.strip()
                if s_stripped:
                    segments.extend(self._split_oversized_text(s_stripped, page_number))
            return segments

        # If a single sentence is still larger than chunk_size, split by clauses
        clauses = self.CLAUSE_SPLIT_REGEX.split(text)
        if len(clauses) > 1:
            segments = []
            for c in clauses:
                c_stripped = c.strip()
                if c_stripped:
                    segments.extend(self._split_oversized_text(c_stripped, page_number))
            return segments

        # If still oversized, split by words
        words = text.split()
        if len(words) > 1:
            segments = []
            current_words: List[str] = []
            current_len = 0
            for w in words:
                word_add = len(w) + (1 if current_words else 0)
                if current_len + word_add <= max_size:
                    current_words.append(w)
                    current_len += word_add
                else:
                    if current_words:
                        segments.append(TextSegment(text=" ".join(current_words), page_number=page_number))
                    current_words = [w]
                    current_len = len(w)
            if current_words:
                segments.append(TextSegment(text=" ".join(current_words), page_number=page_number))
            return segments

        # Extreme fallback: single unbroken string longer than chunk_size
        return [
            TextSegment(text=text[i : i + max_size], page_number=page_number)
            for i in range(0, len(text), max_size)
        ]

    def _extract_segments(
        self,
        text: str,
        pages: Optional[List[ParsedPage]] = None,
    ) -> List[TextSegment]:
        """Converts raw text or parsed pages into hierarchical text segments."""
        segments: List[TextSegment] = []

        if pages:
            for page in pages:
                page_text = page.text.strip()
                if not page_text:
                    continue
                # Split page into paragraphs
                paragraphs = re.split(r"\n\s*\n+", page_text)
                for para in paragraphs:
                    para_clean = para.strip()
                    if not para_clean:
                        continue
                    segments.extend(self._split_oversized_text(para_clean, page.page_number))
        else:
            clean_text = text.strip()
            if not clean_text:
                return []
            paragraphs = re.split(r"\n\s*\n+", clean_text)
            for para in paragraphs:
                para_clean = para.strip()
                if not para_clean:
                    continue
                segments.extend(self._split_oversized_text(para_clean, None))

        return segments

    def _get_overlap_segments(self, units: List[TextSegment]) -> List[TextSegment]:
        """Calculates trailing overlap segments bounded strictly by chunk_overlap."""
        if not units or self.config.chunk_overlap <= 0:
            return []

        overlap_units: List[TextSegment] = []
        accumulated_len = 0

        for unit in reversed(units):
            added_len = len(unit.text) + (2 if overlap_units else 0)
            if accumulated_len + added_len <= self.config.chunk_overlap:
                overlap_units.insert(0, unit)
                accumulated_len += added_len
            else:
                # If no whole unit fit because the last unit is larger than chunk_overlap,
                # take trailing words from that unit that fit within chunk_overlap
                if not overlap_units:
                    words = unit.text.split()
                    trailing_words: List[str] = []
                    words_len = 0
                    for w in reversed(words):
                        w_len = len(w) + (1 if trailing_words else 0)
                        if words_len + w_len <= self.config.chunk_overlap:
                            trailing_words.insert(0, w)
                            words_len += w_len
                        else:
                            break
                    if trailing_words:
                        overlap_units.append(
                            TextSegment(
                                text=" ".join(trailing_words),
                                page_number=unit.page_number,
                            )
                        )
                break

        return overlap_units

    def chunk_document(
        self,
        document_id: str,
        text: str,
        pages: Optional[List[ParsedPage]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Chunk]:
        """
        Segments document into ordered chunks respecting structure, size, overlap,
        and page provenance.
        """
        meta = metadata or {}
        segments = self._extract_segments(text=text, pages=pages)

        if not segments:
            return []

        chunks: List[Chunk] = []
        current_units: List[TextSegment] = []
        current_len = 0
        chunk_idx = 0

        i = 0
        while i < len(segments):
            seg = segments[i]
            sep_len = 2 if current_units else 0
            candidate_len = current_len + sep_len + len(seg.text)

            if candidate_len <= self.config.chunk_size:
                current_units.append(seg)
                current_len = candidate_len
                i += 1
            else:
                # Current chunk is full, emit it
                if current_units:
                    chunk_text = "\n\n".join(u.text for u in current_units).strip()
                    pages_in_chunk = [u.page_number for u in current_units if u.page_number is not None]

                    chunks.append(
                        Chunk(
                            chunk_id=f"chk_{document_id}_{chunk_idx:04d}",
                            document_id=document_id,
                            chunk_index=chunk_idx,
                            content=chunk_text,
                            character_count=len(chunk_text),
                            token_count=max(1, len(chunk_text.split())),
                            page_start=min(pages_in_chunk) if pages_in_chunk else None,
                            page_end=max(pages_in_chunk) if pages_in_chunk else None,
                            metadata=meta,
                        )
                    )
                    chunk_idx += 1

                    # Compute overlap from emitted units
                    overlap_units = self._get_overlap_segments(current_units)
                    # Ensure overlap + next segment doesn't exceed chunk_size; trim overlap if needed
                    while overlap_units and (
                        sum(len(u.text) for u in overlap_units)
                        + (2 * len(overlap_units))
                        + len(seg.text)
                        > self.config.chunk_size
                    ):
                        overlap_units.pop(0)

                    current_units = list(overlap_units)
                    current_len = sum(len(u.text) for u in current_units) + (
                        2 * (len(current_units) - 1) if len(current_units) > 1 else 0
                    )
                else:
                    # Single segment was larger than chunk_size (fallback safeguard)
                    current_units.append(seg)
                    current_len = len(seg.text)
                    i += 1

        # Emit remaining units if present
        if current_units:
            chunk_text = "\n\n".join(u.text for u in current_units).strip()
            pages_in_chunk = [u.page_number for u in current_units if u.page_number is not None]

            # Merge with previous chunk only if it is small and doesn't exceed chunk_size
            if (
                chunks
                and len(chunk_text) < self.config.minimum_chunk_size
                and (chunks[-1].character_count + 2 + len(chunk_text)) <= self.config.chunk_size
            ):
                last_chunk = chunks[-1]
                merged_text = f"{last_chunk.content}\n\n{chunk_text}".strip()
                last_chunk.content = merged_text
                last_chunk.character_count = len(merged_text)
                last_chunk.token_count = max(1, len(merged_text.split()))
                if pages_in_chunk:
                    last_chunk.page_end = max(
                        [p for p in [last_chunk.page_end] if p is not None] + pages_in_chunk
                    )
            else:
                chunks.append(
                    Chunk(
                        chunk_id=f"chk_{document_id}_{chunk_idx:04d}",
                        document_id=document_id,
                        chunk_index=chunk_idx,
                        content=chunk_text,
                        character_count=len(chunk_text),
                        token_count=max(1, len(chunk_text.split())),
                        page_start=min(pages_in_chunk) if pages_in_chunk else None,
                        page_end=max(pages_in_chunk) if pages_in_chunk else None,
                        metadata=meta,
                    )
                )

        return chunks


# Alias to satisfy interface naming while being honest about implementation strategy
SemanticChunker = StructureAwareChunker
