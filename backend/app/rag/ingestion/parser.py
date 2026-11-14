from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import pymupdf

from app.core.errors import CorruptedDocumentError, NoExtractableTextError
from app.core.logging import logger


@dataclass
class ParsedPage:
    page_number: int
    text: str


@dataclass
class ExtractionResult:
    page_count: int
    full_text: str
    preview_snippet: str
    pages: List[ParsedPage] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class PDFParser:
    """Extracts text page-by-page from PDF documents using PyMuPDF."""

    @staticmethod
    def parse(file_bytes: bytes, filename: str = "document.pdf") -> ExtractionResult:
        if not file_bytes:
            raise NoExtractableTextError(filename)

        try:
            doc = pymupdf.open(stream=file_bytes, filetype="pdf")
        except Exception as e:
            logger.warning(f"Failed to open PDF stream for {filename}: {e}")
            raise CorruptedDocumentError(filename, reason=f"Invalid or corrupted PDF stream ({str(e)})")

        if doc.needs_pass:
            doc.close()
            raise CorruptedDocumentError(filename, reason="PDF document is password protected.")

        page_count = len(doc)
        if page_count == 0:
            doc.close()
            raise NoExtractableTextError(filename)

        pages: List[ParsedPage] = []
        combined_text_parts: List[str] = []

        try:
            for page_num in range(page_count):
                page = doc[page_num]
                text = page.get_text("text").strip()
                pages.append(ParsedPage(page_number=page_num + 1, text=text))
                if text:
                    combined_text_parts.append(text)

            metadata = {
                "title": doc.metadata.get("title") or "",
                "author": doc.metadata.get("author") or "",
                "subject": doc.metadata.get("subject") or "",
                "creator": doc.metadata.get("creator") or "",
                "producer": doc.metadata.get("producer") or "",
            }
        except Exception as e:
            doc.close()
            logger.warning(f"Error during page extraction for {filename}: {e}")
            raise CorruptedDocumentError(filename, reason=f"Error reading PDF page contents ({str(e)})")
        finally:
            doc.close()

        full_text = "\n\n".join(combined_text_parts).strip()
        if not full_text:
            raise NoExtractableTextError(filename)

        preview = full_text[:1200]
        return ExtractionResult(
            page_count=page_count,
            full_text=full_text,
            preview_snippet=preview,
            pages=pages,
            metadata=metadata,
        )


class TextParser:
    """Extracts plain text with encoding fallback detection."""

    FALLBACK_ENCODINGS = ("utf-8", "utf-8-sig", "latin-1", "cp1252", "utf-16")

    @classmethod
    def parse(cls, file_bytes: bytes, filename: str = "document.txt") -> ExtractionResult:
        if not file_bytes:
            raise NoExtractableTextError(filename)

        decoded_text: Optional[str] = None
        used_encoding: Optional[str] = None

        for enc in cls.FALLBACK_ENCODINGS:
            try:
                decoded_text = file_bytes.decode(enc)
                used_encoding = enc
                break
            except (UnicodeDecodeError, LookupError):
                continue

        if decoded_text is None:
            raise CorruptedDocumentError(filename, reason="Could not decode text using standard encodings.")

        text = decoded_text.strip()
        if not text:
            raise NoExtractableTextError(filename)

        # Virtual page calculation (e.g. ~50 lines per virtual page)
        lines = text.splitlines()
        page_count = max(1, (len(lines) + 49) // 50)
        preview = text[:1200]

        return ExtractionResult(
            page_count=page_count,
            full_text=text,
            preview_snippet=preview,
            pages=[ParsedPage(page_number=1, text=text)],
            metadata={"encoding": used_encoding, "line_count": len(lines)},
        )


class DocumentParser:
    """Unified entrypoint that dispatches to format-specific parsers."""

    @classmethod
    def extract_content(cls, file_bytes: bytes, filename: str, file_type: str) -> ExtractionResult:
        normalized_type = file_type.lower()
        if normalized_type == "pdf":
            return PDFParser.parse(file_bytes, filename=filename)
        elif normalized_type == "txt":
            return TextParser.parse(file_bytes, filename=filename)
        else:
            raise CorruptedDocumentError(filename, reason=f"Unsupported parser type: {file_type}")
