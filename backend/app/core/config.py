from pathlib import Path
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
    CORS_ORIGINS: str = "http://localhost:3000,http://127.0.0.1:3000"

    # Storage & Database paths
    STORAGE_DIR: str = "data/storage"
    MAX_UPLOAD_SIZE_MB: int = 10
    SQLITE_DB_PATH: str = "data/sift_metadata.db"
    VECTOR_DB_DIR: str = "data/chroma"
    CHROMA_COLLECTION_NAME: str = "sift_document_embeddings"

    # RAG chunking defaults
    DEFAULT_EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
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
    def storage_path(self) -> Path:
        path = Path(self.STORAGE_DIR)
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def database_path(self) -> Path:
        path = Path(self.SQLITE_DB_PATH)
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def vector_db_path(self) -> Path:
        path = Path(self.VECTOR_DB_DIR)
        path.mkdir(parents=True, exist_ok=True)
        return path


settings = Settings()
