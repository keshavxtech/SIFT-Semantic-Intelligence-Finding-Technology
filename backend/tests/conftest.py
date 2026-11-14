import os
import shutil
import tempfile
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

# Configure isolated temporary directory for test storage and database inside the workspace
workspace_tmp = Path(__file__).resolve().parent.parent / ".test_tmp"
workspace_tmp.mkdir(exist_ok=True)
temp_dir = tempfile.mkdtemp(prefix="sift_test_env_", dir=str(workspace_tmp))
test_storage = os.path.join(temp_dir, "storage")
test_db = os.path.join(temp_dir, "test_metadata.db")
test_chroma = os.path.join(temp_dir, "chroma")

os.environ["STORAGE_DIR"] = test_storage
os.environ["SQLITE_DB_PATH"] = test_db
os.environ["VECTOR_DB_DIR"] = test_chroma
os.environ["MAX_UPLOAD_SIZE_MB"] = "10"
os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["HF_HUB_OFFLINE"] = "1"

from app.core.config import settings
# Overwrite settings instances with test paths
settings.STORAGE_DIR = test_storage
settings.SQLITE_DB_PATH = test_db
settings.VECTOR_DB_DIR = test_chroma
settings.MAX_UPLOAD_SIZE_MB = 10

from app.main import app
from app.repositories.document_repository import DocumentRepository, document_repository
from app.repositories.chunk_repository import ChunkRepository, chunk_repository
from app.repositories.embedding_repository import EmbeddingRepository, embedding_repository
from app.services.document_service import document_service
from app.services.chunking_service import chunking_service
from app.rag.embeddings.base import MockEmbeddingProvider
from app.rag.embeddings.embedding_service import embedding_service

# Point default singletons to test storage/db
test_repo = DocumentRepository(db_path=Path(test_db))
document_repository._db_path = Path(test_db)
document_repository._init_db()
document_service.repo = test_repo

test_chunk_repo = ChunkRepository(db_path=Path(test_db))
chunk_repository._db_path = Path(test_db)
chunk_repository._init_db()
chunking_service.doc_repo = test_repo
chunking_service.chunk_repo = test_chunk_repo

test_embed_repo = EmbeddingRepository(db_path=Path(test_db))
embedding_repository._db_path = Path(test_db)
embedding_repository._init_db()
embedding_service.doc_repo = test_repo
embedding_service.chunk_svc = chunking_service
embedding_service.embed_repo = test_embed_repo
embedding_service.provider = MockEmbeddingProvider()

from app.rag.retrieval.vector_store import vector_store
from app.rag.retrieval.retriever import semantic_retriever
from app.services.indexing_service import indexing_service
from app.services.search_service import search_service

vector_store.persist_directory = Path(test_chroma)
vector_store.collection_name = "test_sift_embeddings"
vector_store._client = None
vector_store._collection = None

semantic_retriever.vector_store = vector_store
semantic_retriever._custom_provider = MockEmbeddingProvider()

indexing_service.doc_repo = test_repo
indexing_service.chunk_repo = test_chunk_repo
indexing_service.embed_repo = test_embed_repo
indexing_service.vector_store = vector_store

search_service.retriever = semantic_retriever
search_service.doc_repo = test_repo

from app.repositories.chat_repository import ChatRepository, chat_repository
from app.rag.generation.mock_provider import MockLLMProvider
from app.services.chat_service import chat_service

test_chat_repo = ChatRepository(db_path=Path(test_db))
chat_repository._db_path = Path(test_db)
chat_repository._init_db()

chat_service.retriever = semantic_retriever
chat_service.chat_repo = test_chat_repo
chat_service.doc_repo = test_repo
chat_service.provider = MockLLMProvider()


@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    yield
    # Cleanup temp directory after test session
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client
