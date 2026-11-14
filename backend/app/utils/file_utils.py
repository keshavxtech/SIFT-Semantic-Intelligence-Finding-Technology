import hashlib
import re
from pathlib import Path
from typing import Optional, Tuple


ALLOWED_EXTENSIONS = {
    ".pdf": "pdf",
    ".txt": "txt",
}

ALLOWED_MIME_TYPES = {
    "pdf": {"application/pdf", "application/x-pdf", "application/octet-stream"},
    "txt": {"text/plain", "text/csv", "application/octet-stream", "text/markdown"},
}


def sanitize_filename(filename: str) -> str:
    """
    Sanitizes an uploaded filename to prevent directory traversal and filesystem attacks.
    Preserves clean alphanumeric characters, hyphens, underscores, and dots.
    """
    if not filename:
        return "untitled_document"

    # Extract base name to eliminate directory paths (both POSIX and Windows)
    clean_name = filename.replace("\\", "/").split("/")[-1].strip()

    # Split name and extension
    path_obj = Path(clean_name)
    stem = path_obj.stem
    ext = path_obj.suffix.lower()

    # Remove non-whitelisted characters from stem
    sanitized_stem = re.sub(r"[^a-zA-Z0-9_\-\.]", "_", stem)
    # Collapse consecutive underscores or dots
    sanitized_stem = re.sub(r"_+", "_", sanitized_stem)
    sanitized_stem = sanitized_stem.strip("._-")

    if not sanitized_stem:
        sanitized_stem = "document"

    # Enforce reasonable stem length
    sanitized_stem = sanitized_stem[:100]

    return f"{sanitized_stem}{ext}"


def validate_file_extension(filename: str) -> Tuple[bool, str]:
    """Validates if filename has an allowed extension (.pdf, .txt)."""
    ext = Path(filename).suffix.lower()
    if ext in ALLOWED_EXTENSIONS:
        return True, ALLOWED_EXTENSIONS[ext]
    return False, ext


def validate_mime_type(content_type: Optional[str], ext: str) -> bool:
    """
    Validates MIME type against the file extension.
    Permits application/octet-stream as common browser fallback.
    """
    if not content_type:
        return True  # If client did not provide MIME type, extension validation takes precedence

    normalized_type = content_type.lower().split(";")[0].strip()
    valid_types = ALLOWED_MIME_TYPES.get(ext, set())
    return normalized_type in valid_types


def is_valid_pdf_signature(content: bytes) -> bool:
    """Checks for standard PDF file magic bytes (%PDF-)."""
    return len(content) >= 5 and content[:5] == b"%PDF-"


def compute_sha256(data: bytes) -> str:
    """Computes a truncated SHA256 checksum for deduplication and integrity tracking."""
    hasher = hashlib.sha256()
    hasher.update(data)
    return f"sha256:{hasher.hexdigest()[:16]}"


def format_bytes_to_human(num_bytes: int) -> str:
    """Formats raw bytes into a human-readable size string."""
    if num_bytes >= 1048576:
        return f"{num_bytes / 1048576:.1f} MB"
    return f"{max(1, round(num_bytes / 1024))} KB"
