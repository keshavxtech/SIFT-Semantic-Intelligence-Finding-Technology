import os
from pathlib import Path
import tempfile
from typing import List, Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    PROJECT_NAME: str = "SIFT Document Intelligence API"
    VERSION: str = "0.1.0"
    ENVIRONMENT: str = "development"
    DEBUG: bool = True

    # Server settings
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    API_V1_PREFIX: str = "/api"

    # CORS
    CORS_ORIGINS: str = "http://localhost:3000,http://127.0.0.1:3000,https://sift-semantic-intelligence-finding.vercel.app"

    # Storage & Database paths
    STORAGE_DIR: str = "data/storage"
    MAX_UPLOAD_SIZE_MB: int = 10
    SQLITE_DB_PATH: str = "data/sift_metadata.db"
    VECTOR_DB_DIR: str = "data/chroma"
    CHROMA_COLLECTION_NAME: str = "sift_document_embeddings"

    # Database provider settings
    DATABASE_PROVIDER: str = "sqlite"  # "sqlite" or "turso"
    TURSO_DATABASE_URL: Optional[str] = None
    TURSO_AUTH_TOKEN: Optional[str] = None

    # Vector store provider settings
    VECTOR_STORE_PROVIDER: str = "chroma"  # "chroma", "pinecone", or "mock"
    PINECONE_API_KEY: Optional[str] = None
    PINECONE_INDEX_NAME: str = "sift-index"
    PINECONE_NAMESPACE: str = "sift-embeddings"
    PINECONE_HOST: Optional[str] = None

    # Embedding Provider settings
    EMBEDDING_PROVIDER: str = "sentence-transformers"  # "sentence-transformers", "huggingface", or "mock"
    DEFAULT_EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    HUGGINGFACE_API_KEY: Optional[str] = None
    HUGGINGFACE_EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    HUGGINGFACE_API_URL: str = "https://router.huggingface.co/hf-inference/models/sentence-transformers/all-MiniLM-L6-v2"
    DEFAULT_CHUNK_SIZE: int = 1000
    DEFAULT_CHUNK_OVERLAP: int = 150
    DEFAULT_MINIMUM_CHUNK_SIZE: int = 100

    # LLM & Generation Provider settings
    LLM_PROVIDER: str = "groq"
    GROQ_API_KEY: Optional[str] = None
    GROQ_MODEL: str = "llama-3.1-8b-instant"
    GROQ_API_URL: str = "https://api.groq.com/openai/v1/chat/completions"
    LLM_MAX_TOKENS: int = 1024
    LLM_TEMPERATURE: float = 0.1
    CHAT_HISTORY_LIMIT: int = 10
    RAG_MAX_CHUNKS: int = 5

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @property
    def cors_origins_list(self) -> List[str]:
        return [origin.strip() for origin in self.CORS_ORIGINS.split(",") if origin.strip()]

    @property
    def is_serverless(self) -> bool:
        return bool(os.environ.get("VERCEL") or os.environ.get("AWS_LAMBDA_FUNCTION_NAME"))

    @property
    def is_production(self) -> bool:
        return self.ENVIRONMENT.lower() in ("production", "prod") or self.is_serverless

    @property
    def storage_path(self) -> Path:
        # In serverless/production runtime, relative paths are on a read-only filesystem.
        # Fall back to system temporary directory (/tmp on Linux/Vercel) when default relative path is used.
        if self.is_serverless and self.STORAGE_DIR == "data/storage":
            path = Path(tempfile.gettempdir()) / "sift" / "storage"
        else:
            path = Path(self.STORAGE_DIR)
        try:
            path.mkdir(parents=True, exist_ok=True)
        except OSError:
            pass
        return path

    def validate_production_configuration(self) -> None:
        """
        Validates that production environments do not silently fall back
        to unsupported local storage or local ML libraries.
        """
        if not self.is_production:
            return

        # 1. Database validation
        if self.DATABASE_PROVIDER.lower() != "turso":
            raise ValueError(
                f"Production deployment requires hosted database provider 'turso' (got '{self.DATABASE_PROVIDER}'). "
                "Local SQLite is not supported in serverless production."
            )
        if not self.TURSO_DATABASE_URL or not self.TURSO_AUTH_TOKEN:
            raise ValueError(
                "TURSO_DATABASE_URL and TURSO_AUTH_TOKEN must be configured when running in production."
            )

        # 2. Vector store validation
        if self.VECTOR_STORE_PROVIDER.lower() != "pinecone":
            raise ValueError(
                f"Production deployment requires hosted vector store provider 'pinecone' (got '{self.VECTOR_STORE_PROVIDER}'). "
                "ChromaDB is not installed in production."
            )
        if not self.PINECONE_API_KEY:
            raise ValueError(
                "PINECONE_API_KEY must be configured when running in production."
            )

        # 3. Embedding provider validation
        if self.EMBEDDING_PROVIDER.lower() not in ("huggingface", "hf", "hosted"):
            raise ValueError(
                f"Production deployment requires hosted embedding provider 'huggingface' (got '{self.EMBEDDING_PROVIDER}'). "
                "SentenceTransformers is not installed in production."
            )
        if not self.HUGGINGFACE_API_KEY:
            raise ValueError(
                "HUGGINGFACE_API_KEY must be configured when running in production."
            )

        # 4. LLM provider validation
        if self.LLM_PROVIDER.lower() == "groq" and not self.GROQ_API_KEY:
            raise ValueError(
                "GROQ_API_KEY must be configured when running in production."
            )

    @property
    def database_path(self) -> Path:
        path = Path(self.SQLITE_DB_PATH)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
        except OSError:
            pass
        return path

    @property
    def vector_db_path(self) -> Path:
        path = Path(self.VECTOR_DB_DIR)
        try:
            path.mkdir(parents=True, exist_ok=True)
        except OSError:
            pass
        return path


settings = Settings()
