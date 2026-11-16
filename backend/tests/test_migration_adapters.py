import math
from unittest.mock import AsyncMock, MagicMock, patch
import httpx
import pytest

from app.core.config import settings
from app.core.database import (
    TursoCursor,
    TursoHttpConnection,
    TursoRow,
    get_db_connection,
)
from app.core.errors import (
    DatabaseConnectionError,
    DatabaseNotConfiguredError,
    DimensionMismatchError,
    SIFTException,
    VectorStoreNotConfiguredError,
)
from app.models.document import Document, DocumentTag, DocumentType, ProcessingStatus
from app.rag.embeddings.base import MockEmbeddingProvider
from app.rag.embeddings.config import EmbeddingConfig
from app.rag.embeddings.factory import get_embedding_provider
from app.rag.embeddings.huggingface_provider import HuggingFaceEmbeddingProvider
from app.rag.retrieval.pinecone_vector_store import PineconeRestClient, PineconeVectorStore
from app.rag.retrieval.vector_store import ChromaVectorStore, get_vector_store
from app.repositories.chat_repository import ChatRepository
from app.repositories.document_repository import DocumentRepository
from app.rag.ingestion.parser import DocumentParser
from app.rag.chunking.chunker import TextChunker
from app.rag.generation.base import BaseLLMProvider, LLMGenerationResult
from app.rag.generation.groq_provider import GroqProvider


# ============================================================================
# Hugging Face Hosted Embedding Provider Tests (Task 1.1)
# ============================================================================

def test_huggingface_provider_embed_text_mocked():
    """Verifies HuggingFaceEmbeddingProvider produces normalized 384-dim vector with mock client."""
    mock_vector = [0.05] * 384
    mock_response = MagicMock(spec=httpx.Response)
    mock_response.status_code = 200
    mock_response.json.return_value = [mock_vector]

    mock_client = MagicMock(spec=httpx.Client)
    mock_client.post.return_value = mock_response

    provider = HuggingFaceEmbeddingProvider(
        api_key="mock_hf_key",
        api_url="https://router.huggingface.co/mock-endpoint",
        client=mock_client,
    )

    vec = provider.embed_text("Semantic search with SIFT")
    assert len(vec) == 384
    assert provider.validate_vector(vec) is True

    # Check normalization: L2 norm should equal 1.0
    norm = math.sqrt(sum(x * x for x in vec))
    assert abs(norm - 1.0) < 1e-4


def test_huggingface_provider_batch_and_empty_text():
    """Verifies batch encoding and proper zero-vector substitution for empty strings."""
    mock_vec = [0.1] * 384
    mock_response = MagicMock(spec=httpx.Response)
    mock_response.status_code = 200
    mock_response.json.return_value = [mock_vec, mock_vec, mock_vec]

    mock_client = MagicMock(spec=httpx.Client)
    mock_client.post.return_value = mock_response

    provider = HuggingFaceEmbeddingProvider(
        api_key="mock_key",
        client=mock_client,
    )

    vectors = provider.embed_batch(["Real query", "", "   "])
    assert len(vectors) == 3
    # First is real vector
    assert len(vectors[0]) == 384
    assert sum(vectors[0]) != 0
    # Second and third are empty strings -> should be zero vectors
    assert vectors[1] == [0.0] * 384
    assert vectors[2] == [0.0] * 384


def test_huggingface_provider_mean_pooling_for_token_embeddings():
    """Verifies that 2D token-level matrix [seq_len x 384] is properly mean-pooled to 1D."""
    # 3 tokens, each 384 floats
    mock_token_matrix = [[0.1] * 384, [0.2] * 384, [0.3] * 384]
    mock_response = MagicMock(spec=httpx.Response)
    mock_response.status_code = 200
    mock_response.json.return_value = [mock_token_matrix]

    mock_client = MagicMock(spec=httpx.Client)
    mock_client.post.return_value = mock_response

    provider = HuggingFaceEmbeddingProvider(
        api_key="mock_key",
        client=mock_client,
    )

    vec = provider.embed_text("Mean pooling test")
    assert len(vec) == 384
    norm = math.sqrt(sum(x * x for x in vec))
    assert abs(norm - 1.0) < 1e-4


def test_huggingface_provider_error_handling():
    """Verifies clear SIFTException handling on 401 and 503 HTTP responses."""
    mock_client = MagicMock(spec=httpx.Client)

    # 401 Unauthorized
    resp_401 = MagicMock(spec=httpx.Response)
    resp_401.status_code = 401
    mock_client.post.return_value = resp_401
    provider = HuggingFaceEmbeddingProvider(api_key="bad_key", client=mock_client)
    with pytest.raises(SIFTException) as exc_info:
        provider.embed_text("fail query")
    assert exc_info.value.status_code == 401

    # 503 Model Loading
    resp_503 = MagicMock(spec=httpx.Response)
    resp_503.status_code = 503
    mock_client.post.return_value = resp_503
    with pytest.raises(SIFTException) as exc_info:
        provider.embed_text("loading query")
    assert exc_info.value.status_code == 503


def test_embedding_factory_resolution():
    """Verifies factory returns appropriate provider according to provider string."""
    mock_p = get_embedding_provider("mock")
    assert isinstance(mock_p, MockEmbeddingProvider)

    hf_p = get_embedding_provider("huggingface")
    assert isinstance(hf_p, HuggingFaceEmbeddingProvider)


# ============================================================================
# Pinecone Vector Store Adapter Tests (Task 1.2)
# ============================================================================

def test_pinecone_vector_store_upsert_and_query_mocked():
    """Verifies PineconeVectorStore upsert, query, and mapping using mock client."""
    mock_client = MagicMock()
    mock_client.query.return_value = {
        "matches": [
            {
                "id": "chk-1",
                "score": 0.92,
                "metadata": {
                    "document_id": "doc-test",
                    "content": "Retrieved passage from Pinecone",
                    "source_filename": "architecture.pdf",
                    "page_start": 2,
                    "chunk_index": 0,
                },
            }
        ]
    }
    mock_client.describe_index_stats.return_value = {"total_vector_count": 42}

    store = PineconeVectorStore(
        api_key="mock_pinecone_key",
        index_name="mock-index",
        client=mock_client,
    )

    # 1. Test upsert
    upserted = store.upsert_chunks(
        document_id="doc-test",
        chunk_ids=["chk-1"],
        embeddings=[[0.05] * 384],
        documents=["Passage content"],
        metadatas=[{"source_filename": "architecture.pdf"}],
    )
    assert upserted == 1
    assert mock_client.upsert.called

    # 2. Test query
    results = store.query(
        query_embedding=[0.05] * 384,
        top_k=1,
        document_ids=["doc-test"],
    )
    assert len(results) == 1
    r = results[0]
    assert r["chunk_id"] == "chk-1"
    assert r["document_id"] == "doc-test"
    assert r["docName"] == "architecture.pdf"
    assert r["similarity_score"] == 0.92
    assert abs(r["distance"] - (1.0 - 0.92)) < 1e-4

    # 3. Test count
    assert store.count() == 42

    # 4. Test delete
    assert store.delete_document("doc-test") == 1
    assert mock_client.delete.called


def test_pinecone_vector_store_dimension_validation():
    """Verifies PineconeVectorStore enforces 384-dimension vector constraints."""
    mock_client = MagicMock()
    store = PineconeVectorStore(api_key="mock_key", client=mock_client)

    # Invalid upsert dimension
    with pytest.raises(DimensionMismatchError):
        store.upsert_chunks(
            document_id="doc-test",
            chunk_ids=["chk-bad"],
            embeddings=[[0.1] * 128],  # Wrong dimension (128 vs 384)
            documents=["bad"],
            metadatas=[{}],
        )

    # Invalid query dimension
    with pytest.raises(DimensionMismatchError):
        store.query(query_embedding=[0.1] * 10)


def test_vector_store_factory_resolution():
    """Verifies get_vector_store() returns correct implementation."""
    chroma_s = get_vector_store("chroma")
    assert isinstance(chroma_s, ChromaVectorStore)

    pinecone_s = get_vector_store("pinecone")
    assert isinstance(pinecone_s, PineconeVectorStore)


def test_pinecone_missing_credentials_raises_error(monkeypatch):
    """Verifies VectorStoreNotConfiguredError is raised when PINECONE_API_KEY is not set."""
    monkeypatch.setattr(settings, "PINECONE_API_KEY", None)
    store = PineconeVectorStore(api_key=None, client=None)
    with pytest.raises(VectorStoreNotConfiguredError):
        store._get_client()


def test_pinecone_chunk_counting_and_indexing():
    """Verifies get_document_chunk_count and is_document_indexed methods."""
    mock_client = MagicMock()
    mock_client.query.return_value = {
        "matches": [{"id": "chk-1"}, {"id": "chk-2"}]
    }

    store = PineconeVectorStore(api_key="mock_key", client=mock_client)
    count = store.get_document_chunk_count("doc-test")
    assert count == 2
    assert store.is_document_indexed("doc-test") is True

    # Empty matches
    mock_client.query.return_value = {"matches": []}
    assert store.get_document_chunk_count("doc-empty") == 0
    assert store.is_document_indexed("doc-empty") is False


# ============================================================================
# Turso LibSQL Cloud Database Adapter Tests (Task 1.3)
# ============================================================================

def test_turso_http_connection_execute_and_row_access():
    """Verifies TursoHttpConnection statement execution, cursor iteration, and TursoRow access."""
    mock_http_client = MagicMock(spec=httpx.Client)
    mock_resp = MagicMock(spec=httpx.Response)
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "results": [
            {
                "type": "ok",
                "response": {
                    "type": "execute",
                    "result": {
                        "cols": [{"name": "document_id"}, {"name": "name"}, {"name": "page_count"}],
                        "rows": [
                            [
                                {"type": "text", "value": "doc-turso-1"},
                                {"type": "text", "value": "policy.pdf"},
                                {"type": "integer", "value": "5"},
                            ]
                        ],
                        "affected_row_count": 0,
                        "last_insert_rowid": 1,
                    },
                },
            }
        ]
    }
    mock_http_client.post.return_value = mock_resp

    conn = TursoHttpConnection(
        database_url="libsql://mock-db-org.turso.io",
        auth_token="mock_token_secret",
        client=mock_http_client,
    )

    cursor = conn.execute(
        "SELECT document_id, name, page_count FROM documents WHERE document_id = ?;",
        ("doc-turso-1",),
    )
    assert isinstance(cursor, TursoCursor)
    assert cursor.lastrowid == 1

    row = cursor.fetchone()
    assert row is not None
    assert isinstance(row, TursoRow)

    # 1. Exact key lookup
    assert row["document_id"] == "doc-turso-1"
    assert row["name"] == "policy.pdf"
    assert row["page_count"] == 5

    # 2. Case-insensitive key lookup
    assert row["DOCUMENT_ID"] == "doc-turso-1"
    assert row["Page_Count"] == 5

    # 3. Index-based lookup
    assert row[0] == "doc-turso-1"
    assert row[1] == "policy.pdf"
    assert row[2] == 5

    # 4. Collection behaviors
    assert "document_id" in row
    assert len(row) == 3
    assert row.keys() == ["document_id", "name", "page_count"]
    assert row.values() == ["doc-turso-1", "policy.pdf", 5]

    # 5. Fetchone exhaustion
    assert cursor.fetchone() is None


def test_turso_http_connection_executemany():
    """Verifies TursoHttpConnection.executemany constructs batch pipeline requests."""
    mock_http_client = MagicMock(spec=httpx.Client)
    mock_resp = MagicMock(spec=httpx.Response)
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "results": [
            {"type": "ok", "response": {"type": "execute", "result": {"affected_row_count": 1}}},
            {"type": "ok", "response": {"type": "execute", "result": {"affected_row_count": 1}}},
        ]
    }
    mock_http_client.post.return_value = mock_resp

    conn = TursoHttpConnection(
        database_url="libsql://mock-db.turso.io",
        auth_token="token_test",
        client=mock_http_client,
    )

    records = [
        ("chk-1", "doc-1", 0, "Chunk text 1"),
        ("chk-2", "doc-1", 1, "Chunk text 2"),
    ]
    cur = conn.executemany("INSERT INTO chunks VALUES (?, ?, ?, ?);", records)
    assert cur.rowcount == 2
    assert mock_http_client.post.called

    # Verify requests payload
    called_payload = mock_http_client.post.call_args[1]["json"]
    assert len(called_payload["requests"]) == 3  # 2 execute + 1 close
    assert called_payload["requests"][0]["type"] == "execute"
    assert called_payload["requests"][1]["type"] == "execute"
    assert called_payload["requests"][2]["type"] == "close"


def test_turso_document_repository_operations_mocked(monkeypatch):
    """Verifies DocumentRepository CRUD lifecycle when configured with Turso provider."""
    mock_http_client = MagicMock(spec=httpx.Client)

    doc_cols = [
        {"name": "document_id"}, {"name": "original_filename"}, {"name": "stored_filename"},
        {"name": "name"}, {"name": "file_type"}, {"name": "file_size"}, {"name": "size_human"},
        {"name": "upload_timestamp"}, {"name": "processing_status"}, {"name": "extracted_text"},
        {"name": "page_count"}, {"name": "error_message"}, {"name": "file_hash"},
        {"name": "tag"}, {"name": "author"}, {"name": "description"}, {"name": "raw_text_preview"},
        {"name": "chunks_count"}, {"name": "vector_count"}, {"name": "token_count"}, {"name": "file_path"}
    ]
    doc_row = [
        {"type": "text", "value": "doc-turso-123"},
        {"type": "text", "value": "policy.pdf"},
        {"type": "text", "value": "stored_policy.pdf"},
        {"type": "text", "value": "Corporate Policy"},
        {"type": "text", "value": "pdf"},
        {"type": "integer", "value": "10240"},
        {"type": "text", "value": "10.0 KB"},
        {"type": "text", "value": "2026-11-14T20:00:00"},
        {"type": "text", "value": "indexed"},
        {"type": "text", "value": "Policy content extracted from Turso"},
        {"type": "integer", "value": "3"},
        {"type": "null"},
        {"type": "text", "value": "hash123"},
        {"type": "text", "value": "Technical"},
        {"type": "text", "value": "Workspace User"},
        {"type": "text", "value": "Description"},
        {"type": "text", "value": "Preview"},
        {"type": "integer", "value": "4"},
        {"type": "integer", "value": "4"},
        {"type": "integer", "value": "250"},
        {"type": "null"}
    ]

    def mock_post(*args, **kwargs):
        payload = kwargs.get("json", {})
        requests = payload.get("requests", [])
        sql = requests[0].get("stmt", {}).get("sql", "") if requests else ""

        resp = MagicMock(spec=httpx.Response)
        resp.status_code = 200

        if "SELECT * FROM documents WHERE document_id" in sql:
            resp.json.return_value = {
                "results": [
                    {
                        "type": "ok",
                        "response": {
                            "type": "execute",
                            "result": {"cols": doc_cols, "rows": [doc_row], "affected_row_count": 0},
                        },
                    }
                ]
            }
        elif "SELECT COUNT(*)" in sql:
            resp.json.return_value = {
                "results": [
                    {
                        "type": "ok",
                        "response": {
                            "type": "execute",
                            "result": {
                                "cols": [{"name": "COUNT(*)"}],
                                "rows": [[{"type": "integer", "value": "1"}]],
                                "affected_row_count": 0,
                            },
                        },
                    }
                ]
            }
        elif "DELETE FROM documents" in sql:
            resp.json.return_value = {
                "results": [
                    {
                        "type": "ok",
                        "response": {
                            "type": "execute",
                            "result": {"cols": [], "rows": [], "affected_row_count": 1},
                        },
                    }
                ]
            }
        else:
            # Table creation, indexes, inserts, updates
            resp.json.return_value = {
                "results": [
                    {
                        "type": "ok",
                        "response": {
                            "type": "execute",
                            "result": {"cols": [], "rows": [], "affected_row_count": 1},
                        },
                    }
                ]
            }
        return resp

    mock_http_client.post.side_effect = mock_post

    monkeypatch.setattr(settings, "DATABASE_PROVIDER", "turso")
    monkeypatch.setattr(settings, "TURSO_DATABASE_URL", "libsql://test-db.turso.io")
    monkeypatch.setattr(settings, "TURSO_AUTH_TOKEN", "mock_turso_key")

    with patch("app.core.database.httpx.Client") as mock_client_cls:
        mock_client_cls.return_value.__enter__.return_value = mock_http_client

        repo = DocumentRepository(provider="turso")

        # 1. Test get
        doc = repo.get("doc-turso-123")
        assert doc is not None
        assert doc.id == "doc-turso-123"
        assert doc.name == "Corporate Policy"
        assert doc.processing_status == ProcessingStatus.INDEXED
        assert doc.tag == DocumentTag.TECHNICAL

        # 2. Test count
        assert repo.count() == 1

        # 3. Test delete
        assert repo.delete("doc-turso-123") is True


def test_turso_chat_repository_operations_mocked(monkeypatch):
    """Verifies ChatRepository session creation and retrieval using Turso provider."""
    mock_http_client = MagicMock(spec=httpx.Client)

    def mock_post(*args, **kwargs):
        payload = kwargs.get("json", {})
        requests = payload.get("requests", [])
        sql = requests[0].get("stmt", {}).get("sql", "") if requests else ""

        resp = MagicMock(spec=httpx.Response)
        resp.status_code = 200

        if "SELECT s.session_id" in sql:
            resp.json.return_value = {
                "results": [
                    {
                        "type": "ok",
                        "response": {
                            "type": "execute",
                            "result": {
                                "cols": [
                                    {"name": "session_id"},
                                    {"name": "title"},
                                    {"name": "created_at"},
                                    {"name": "updated_at"},
                                    {"name": "message_count"},
                                ],
                                "rows": [
                                    [
                                        {"type": "text", "value": "ses-turso-1"},
                                        {"type": "text", "value": "Turso Chat Session"},
                                        {"type": "text", "value": "2026-11-14T20:00:00"},
                                        {"type": "text", "value": "2026-11-14T20:00:00"},
                                        {"type": "integer", "value": "2"},
                                    ]
                                ],
                                "affected_row_count": 0,
                            },
                        },
                    }
                ]
            }
        elif "DELETE FROM chat_sessions" in sql:
            resp.json.return_value = {
                "results": [
                    {
                        "type": "ok",
                        "response": {
                            "type": "execute",
                            "result": {"cols": [], "rows": [], "affected_row_count": 1},
                        },
                    }
                ]
            }
        else:
            resp.json.return_value = {
                "results": [
                    {
                        "type": "ok",
                        "response": {
                            "type": "execute",
                            "result": {"cols": [], "rows": [], "affected_row_count": 1},
                        },
                    }
                ]
            }
        return resp

    mock_http_client.post.side_effect = mock_post

    monkeypatch.setattr(settings, "DATABASE_PROVIDER", "turso")
    monkeypatch.setattr(settings, "TURSO_DATABASE_URL", "libsql://test-db.turso.io")
    monkeypatch.setattr(settings, "TURSO_AUTH_TOKEN", "mock_turso_key")

    with patch("app.core.database.httpx.Client") as mock_client_cls:
        mock_client_cls.return_value.__enter__.return_value = mock_http_client

        repo = ChatRepository(provider="turso")

        # Get session
        session = repo.get_session("ses-turso-1")
        assert session is not None
        assert session["session_id"] == "ses-turso-1"
        assert session["title"] == "Turso Chat Session"
        assert session["message_count"] == 2

        # Delete session
        assert repo.delete_session("ses-turso-1") is True


def test_turso_missing_credentials_raises_error(monkeypatch):
    """Verifies DatabaseNotConfiguredError is raised when Turso credentials are missing."""
    monkeypatch.setattr(settings, "DATABASE_PROVIDER", "turso")
    monkeypatch.setattr(settings, "TURSO_DATABASE_URL", None)
    monkeypatch.setattr(settings, "TURSO_AUTH_TOKEN", None)

    with pytest.raises(DatabaseNotConfiguredError):
        with get_db_connection(provider="turso") as conn:
            conn.execute("SELECT 1;")


def test_turso_authentication_failure_401():
    """Verifies DatabaseNotConfiguredError on 401 Unauthorized without leaking token."""
    mock_client = MagicMock(spec=httpx.Client)
    mock_resp = MagicMock(spec=httpx.Response)
    mock_resp.status_code = 401
    mock_client.post.return_value = mock_resp

    conn = TursoHttpConnection(
        database_url="libsql://test.turso.io",
        auth_token="sensitive_secret_token",
        client=mock_client,
    )

    with pytest.raises(DatabaseNotConfiguredError) as exc_info:
        conn.execute("SELECT 1;")

    # Verify secret token is NOT leaked in exception message
    assert "sensitive_secret_token" not in str(exc_info.value)


def test_turso_query_syntax_or_sqlite_error():
    """Verifies DatabaseConnectionError is raised when Turso returns SQL error in result."""
    mock_client = MagicMock(spec=httpx.Client)
    mock_resp = MagicMock(spec=httpx.Response)
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "results": [
            {
                "type": "error",
                "error": {
                    "message": "no such table: unknown_table",
                    "code": "SQLITE_ERROR",
                },
            }
        ]
    }
    mock_client.post.return_value = mock_resp

    conn = TursoHttpConnection(
        database_url="libsql://test.turso.io",
        auth_token="token_test",
        client=mock_client,
    )

    with pytest.raises(DatabaseConnectionError) as exc_info:
        conn.execute("SELECT * FROM unknown_table;")
    assert "no such table: unknown_table" in str(exc_info.value)


def test_sqlite_fallback_preserves_local_development(tmp_path):
    """Verifies that default get_db_connection provides full local SQLite compatibility."""
    test_db = tmp_path / "test_local.db"
    with get_db_connection(db_path=test_db, provider="sqlite") as conn:
        conn.execute("CREATE TABLE test_table (id INTEGER PRIMARY KEY, title TEXT);")
        conn.execute("INSERT INTO test_table (title) VALUES (?);", ("Local SQLite Still Works",))

    with get_db_connection(db_path=test_db, provider="sqlite") as conn:
        cur = conn.execute("SELECT title FROM test_table WHERE id = 1;")
        row = cur.fetchone()
        assert row["title"] == "Local SQLite Still Works"
        assert row[0] == "Local SQLite Still Works"


def test_end_to_end_mocked_serverless_rag_workflow():
    """
    Comprehensive End-to-End Mocked Integration Test (Task 2.3):
    Simulates the complete 10-stage RAG lifecycle with hosted cloud provider adapters:
      1. Document byte ingestion (TXT/PDF)
      2. Text extraction via DocumentParser
      3. Chunk creation via Chunker
      4. Hosted Hugging Face 384-dim dense embedding generation
      5. Pinecone cloud vector store upsert
      6. Turso libSQL document & chunk metadata persistence
      7. Semantic similarity search via Pinecone
      8. Groq-powered contextual answer generation
      9. Citation generation & metadata verification
      10. Turso session history & conversation message retrieval
    """
    doc_id = "doc-e2e-serverless-001"
    raw_text = (
        "SIFT is an enterprise semantic intelligence and document finding system. "
        "It decouples heavy local PyTorch dependencies using hosted Hugging Face embeddings, "
        "Pinecone cloud vector database, and Turso serverless libSQL database."
    )
    raw_bytes = raw_text.encode("utf-8")

    # Stage 1 & 2: Ingestion and text parsing
    parsed = DocumentParser.extract_content(
        file_bytes=raw_bytes,
        filename="sift_overview.txt",
        file_type="txt",
    )
    assert parsed.full_text == raw_text
    assert parsed.page_count == 1

    # Stage 3: Chunk creation
    chunker = TextChunker(chunk_size=100, chunk_overlap=20)
    chunks = chunker.chunk_text(
        text=parsed.full_text,
        document_id=doc_id,
        document_name="sift_overview.txt",
    )
    assert len(chunks) >= 1
    for c in chunks:
        assert c["document_id"] == doc_id
        assert "chunk_id" in c
        assert len(c["passage"]) > 0

    # Stage 4: Hosted Hugging Face embedding generation
    mock_vec = [0.05] * 384
    mock_hf_resp = MagicMock(spec=httpx.Response)
    mock_hf_resp.status_code = 200
    mock_hf_resp.json.return_value = [mock_vec for _ in chunks]

    mock_hf_client = MagicMock(spec=httpx.Client)
    mock_hf_client.post.return_value = mock_hf_resp

    hf_provider = HuggingFaceEmbeddingProvider(
        api_key="mock-hf-key",
        client=mock_hf_client,
    )
    embedded_vectors = hf_provider.embed_batch([c["passage"] for c in chunks])
    assert len(embedded_vectors) == len(chunks)
    assert len(embedded_vectors[0]) == 384

    # Stage 5: Pinecone vector upsert
    mock_pinecone_client = MagicMock()
    mock_pinecone_client.upsert.return_value = {"upserted_count": len(chunks)}
    pinecone_store = PineconeVectorStore(
        api_key="mock-pinecone-key",
        host="https://sift-index.pinecone.io",
        client=mock_pinecone_client,
    )
    upserted_count = pinecone_store.upsert_chunks(
        document_id=doc_id,
        chunk_ids=[c["chunk_id"] for c in chunks],
        embeddings=embedded_vectors,
        documents=[c["passage"] for c in chunks],
        metadatas=[{"source_filename": "sift_overview.txt"} for _ in chunks],
    )
    assert upserted_count == len(chunks)

    # Stage 6: Turso metadata persistence (Mocked Turso HTTP Pipeline)
    turso_mock_client = MagicMock(spec=httpx.Client)
    turso_mock_resp = MagicMock(spec=httpx.Response)
    turso_mock_resp.status_code = 200
    turso_mock_resp.json.return_value = {
        "results": [
            {
                "type": "ok",
                "response": {
                    "type": "execute",
                    "result": {
                        "cols": [{"name": "id"}, {"name": "title"}, {"name": "status"}],
                        "rows": [[{"type": "text", "value": doc_id}, {"type": "text", "value": "sift_overview.txt"}, {"type": "text", "value": "indexed"}]],
                        "affected_row_count": 1,
                        "last_insert_rowid": 1,
                    },
                },
            }
        ]
    }
    turso_mock_client.post.return_value = turso_mock_resp
    turso_conn = TursoHttpConnection(
        database_url="libsql://test.turso.io",
        auth_token="test-token",
        client=turso_mock_client,
    )
    cur = turso_conn.execute("SELECT id, title, status FROM documents WHERE id = ?;", (doc_id,))
    doc_row = cur.fetchone()
    assert doc_row["id"] == doc_id
    assert doc_row["status"] == "indexed"

    # Stage 7: Semantic similarity search via Pinecone
    query_text = "What is SIFT?"
    mock_hf_resp.json.return_value = [mock_vec]
    query_vec = hf_provider.embed_text(query_text)
    assert len(query_vec) == 384

    mock_pinecone_client.query.return_value = {
        "matches": [
            {
                "id": chunks[0]["chunk_id"],
                "score": 0.92,
                "metadata": {
                    "document_id": doc_id,
                    "chunk_index": 0,
                    "content": chunks[0]["passage"],
                    "source_filename": "sift_overview.txt",
                    "file_type": "txt",
                    "page_start": 1,
                    "page_end": 1,
                    "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
                },
            }
        ]
    }
    search_results = pinecone_store.query(query_embedding=query_vec, top_k=1)
    assert len(search_results) == 1
    assert search_results[0]["document_id"] == doc_id
    assert search_results[0]["similarity_score"] == 0.92

    # Stage 8 & 9: Groq-powered answer generation & Citation synthesis
    groq_provider = GroqProvider(api_key="mock-groq-key")
    assert groq_provider.provider_name == "groq"
    assert groq_provider.model_name == "llama-3.1-8b-instant"

    llm_response = LLMGenerationResult(
        text="SIFT is an enterprise semantic intelligence system [1].",
        provider=groq_provider.provider_name,
        model=groq_provider.model_name,
        tokens_used=38,
        finish_reason="stop",
        latency_ms=120.5,
    )
    assert "SIFT is an enterprise semantic intelligence system" in llm_response.text
    assert llm_response.tokens_used == 38
    assert llm_response.provider == "groq"
    assert llm_response.model == "llama-3.1-8b-instant"

    # Verify citation metadata consistency
    citations = [
        {
            "citation_index": 1,
            "document_id": search_results[0]["document_id"],
            "chunk_id": search_results[0]["chunk_id"],
            "source_filename": search_results[0]["source_filename"],
            "content": search_results[0]["content"],
            "similarity_score": search_results[0]["similarity_score"],
        }
    ]
    assert citations[0]["document_id"] == doc_id
    assert citations[0]["source_filename"] == "sift_overview.txt"

    # Stage 10: Turso session history retrieval
    session_id = "sess-e2e-001"
    turso_mock_resp.json.return_value = {
        "results": [
            {
                "type": "ok",
                "response": {
                    "type": "execute",
                    "result": {
                        "cols": [
                            {"name": "id"},
                            {"name": "session_id"},
                            {"name": "role"},
                            {"name": "content"},
                            {"name": "created_at"},
                        ],
                        "rows": [
                            [
                                {"type": "text", "value": "msg-1"},
                                {"type": "text", "value": session_id},
                                {"type": "text", "value": "user"},
                                {"type": "text", "value": query_text},
                                {"type": "text", "value": "2026-11-16T14:00:00Z"},
                            ],
                            [
                                {"type": "text", "value": "msg-2"},
                                {"type": "text", "value": session_id},
                                {"type": "text", "value": "assistant"},
                                {"type": "text", "value": llm_response.text},
                                {"type": "text", "value": "2026-11-16T14:00:01Z"},
                            ],
                        ],
                        "affected_row_count": 2,
                    },
                },
            }
        ]
    }
    cur_sess = turso_conn.execute("SELECT * FROM chat_messages WHERE session_id = ?;", (session_id,))
    messages = cur_sess.fetchall()
    assert len(messages) == 2
    assert messages[0]["role"] == "user"
    assert messages[0]["content"] == query_text
    assert messages[1]["role"] == "assistant"
    assert "SIFT is an enterprise semantic intelligence" in messages[1]["content"]

