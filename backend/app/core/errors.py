from typing import Any, Dict, Optional
from fastapi import Request, status
from fastapi.responses import JSONResponse


class SIFTException(Exception):
    def __init__(
        self,
        message: str,
        status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR,
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.details = details or {}


class DocumentNotFoundError(SIFTException):
    def __init__(self, document_id: str):
        super().__init__(
            message=f"Document with ID '{document_id}' was not found.",
            status_code=status.HTTP_404_NOT_FOUND,
            details={"document_id": document_id},
        )


class UnsupportedFileTypeError(SIFTException):
    def __init__(self, extension: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=f"File extension '{extension}' is not supported. Supported extensions: .pdf, .txt",
            status_code=status.HTTP_400_BAD_REQUEST,
            details=details or {"extension": extension, "supported": [".pdf", ".txt"]},
        )


class EmptyFileError(SIFTException):
    def __init__(self, filename: str):
        super().__init__(
            message=f"Uploaded file '{filename}' is empty (0 bytes).",
            status_code=status.HTTP_400_BAD_REQUEST,
            details={"filename": filename, "reason": "empty_file"},
        )


class FileSizeLimitExceededError(SIFTException):
    def __init__(self, size_mb: float, max_mb: int):
        super().__init__(
            message=f"File size ({size_mb:.2f} MB) exceeds maximum allowed limit of {max_mb} MB.",
            status_code=getattr(status, "HTTP_413_CONTENT_TOO_LARGE", 413),
            details={"file_size_mb": size_mb, "max_limit_mb": max_mb},
        )


class CorruptedDocumentError(SIFTException):
    def __init__(self, filename: str, reason: str = "Invalid file structure or unreadable contents."):
        super().__init__(
            message=f"File '{filename}' is corrupted and cannot be parsed: {reason}",
            status_code=getattr(status, "HTTP_422_UNPROCESSABLE_CONTENT", 422),
            details={"filename": filename, "reason": reason},
        )


class NoExtractableTextError(SIFTException):
    def __init__(self, filename: str):
        super().__init__(
            message=f"Document '{filename}' contains no extractable or readable text.",
            status_code=getattr(status, "HTTP_422_UNPROCESSABLE_CONTENT", 422),
            details={"filename": filename, "reason": "no_extractable_text"},
        )


class StorageError(SIFTException):
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            details=details or {},
        )


class IndexingError(SIFTException):
    def __init__(
        self,
        message: str,
        status_code: int = status.HTTP_400_BAD_REQUEST,
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            message=message,
            status_code=status_code,
            details=details or {},
        )


class ModelMismatchError(SIFTException):
    def __init__(self, expected: str, actual: str):
        super().__init__(
            message=f"Model mismatch: embeddings were generated with '{actual}', but active collection expects '{expected}'.",
            status_code=status.HTTP_400_BAD_REQUEST,
            details={"expected_model": expected, "actual_model": actual},
        )


class DimensionMismatchError(SIFTException):
    def __init__(self, expected: int, actual: int):
        super().__init__(
            message=f"Embedding dimension mismatch: expected {expected}, got {actual}.",
            status_code=status.HTTP_400_BAD_REQUEST,
            details={"expected_dimension": expected, "actual_dimension": actual},
        )


class LLMProviderError(SIFTException):
    def __init__(
        self,
        message: str,
        status_code: int = status.HTTP_502_BAD_GATEWAY,
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            message=message,
            status_code=status_code,
            details=details or {},
        )


class LLMProviderNotConfiguredError(SIFTException):
    def __init__(
        self,
        message: str = "No LLM provider is configured. Please set GROQ_API_KEY in your environment or backend/.env.",
    ):
        super().__init__(
            message=message,
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            details={"provider": "unconfigured"},
        )


class LLMTimeoutError(SIFTException):
    def __init__(self, timeout_seconds: float):
        super().__init__(
            message=f"LLM generation timed out after {timeout_seconds:.1f}s. Please retry your request.",
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            details={"timeout_seconds": timeout_seconds},
        )


class LLMRateLimitError(SIFTException):
    def __init__(
        self,
        message: str = "LLM provider rate limit exceeded. Please wait a moment before trying again.",
    ):
        super().__init__(
            message=message,
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            details={"retry_after": "standard_quota_delay"},
        )


class EmbeddingProviderNotConfiguredError(SIFTException):
    def __init__(
        self,
        message: str = "Embedding provider is not configured. Please set HUGGINGFACE_API_KEY in your environment or backend/.env.",
    ):
        super().__init__(
            message=message,
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            details={"provider": "unconfigured"},
        )


class VectorStoreNotConfiguredError(SIFTException):
    def __init__(
        self,
        message: str = "Vector store is not configured. Please set PINECONE_API_KEY in your environment or backend/.env.",
    ):
        super().__init__(
            message=message,
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            details={"provider": "unconfigured"},
        )


class DatabaseNotConfiguredError(SIFTException):
    def __init__(
        self,
        message: str = "Database is not configured. Please set TURSO_DATABASE_URL and TURSO_AUTH_TOKEN in your environment or backend/.env.",
    ):
        super().__init__(
            message=message,
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            details={"provider": "turso_unconfigured"},
        )


class DatabaseConnectionError(SIFTException):
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            status_code=status.HTTP_502_BAD_GATEWAY,
            details=details or {},
        )


class ChatSessionNotFoundError(SIFTException):
    def __init__(self, session_id: str):
        super().__init__(
            message=f"Chat session '{session_id}' was not found.",
            status_code=status.HTTP_404_NOT_FOUND,
            details={"session_id": session_id},
        )


async def sift_exception_handler(request: Request, exc: SIFTException) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": True,
            "message": exc.message,
            "details": exc.details,
            "path": str(request.url.path),
        },
    )
