import io
import math
import os
import pytest
from app.rag.chunking.models import Chunk
from app.rag.embeddings.base import BaseEmbeddingProvider, MockEmbeddingProvider
from app.rag.embeddings.config import EmbeddingConfig, DEFAULT_EMBEDDING_CONFIG
from app.rag.embeddings.embedding_service import EmbeddingService
from app.rag.embeddings.models import ChunkEmbedding
from app.rag.embeddings.sentence_transformer_provider import SentenceTransformerProvider
from app.repositories.embedding_repository import EmbeddingRepository


# -------------------------------------------------------------
# Unit Tests for Base & Mock Embedding Provider
# -------------------------------------------------------------

def test_mock_provider_initialization():
    config = EmbeddingConfig(model_name="test-model", dimension=384, batch_size=16)
    provider = MockEmbeddingProvider(config=config)
    assert provider.model_name == "test-model"
    assert provider.dimension == 384
    assert provider.config.batch_size == 16


def test_single_chunk_embedding():
    provider = MockEmbeddingProvider()
    text = "Artificial intelligence system design and vector representations."
    vector = provider.embed_text(text)

    assert isinstance(vector, list)
    assert len(vector) == 384
    assert provider.validate_vector(vector) is True


def test_batch_chunk_embedding():
    provider = MockEmbeddingProvider()
    texts = [
        "First technical document passage.",
        "Second passage regarding database indexing.",
        "Third passage covering enterprise security safeguards.",
    ]
    vectors = provider.embed_batch(texts)

    assert len(vectors) == 3
    for v in vectors:
        assert len(v) == 384
        assert provider.validate_vector(v) is True


def test_embedding_normalization():
    # When normalize_embeddings=True (default), L2 norm must be approximately 1.0
    provider = MockEmbeddingProvider(EmbeddingConfig(normalize_embeddings=True))
    vector = provider.embed_text("Normalization validation test sentence.")
    l2_norm = math.sqrt(sum(x * x for x in vector))
    assert pytest.approx(l2_norm, rel=1e-4) == 1.0


def test_finite_numerical_values():
    provider = MockEmbeddingProvider()
    vector = provider.embed_text("Finite value inspection passage.")
    for val in vector:
        assert isinstance(val, (int, float))
        assert math.isfinite(val)
        assert not math.isnan(val)


def test_empty_input_handling():
    provider = MockEmbeddingProvider()
    # Empty string returns zero vector
    empty_vec = provider.embed_text("")
    assert len(empty_vec) == 384
    assert all(v == 0.0 for v in empty_vec)

    # Empty batch returns empty list
    empty_batch = provider.embed_batch([])
    assert empty_batch == []


def test_deterministic_ordering_and_output():
    provider = MockEmbeddingProvider()
    texts = [f"Deterministic passage index {i}" for i in range(5)]

    run1 = provider.embed_batch(texts)
    run2 = provider.embed_batch(texts)

    assert len(run1) == len(run2) == 5
    for v1, v2 in zip(run1, run2):
        assert v1 == v2


def test_vector_dimension_validation():
    provider = MockEmbeddingProvider()
    # Correct dimension
    assert provider.validate_vector([0.1] * 384) is True
    # Wrong dimension
    assert provider.validate_vector([0.1] * 128) is False
    assert provider.validate_vector([0.1] * 512) is False
    # Non-finite values
    assert provider.validate_vector([float("nan")] * 384) is False
    assert provider.validate_vector([float("inf")] * 384) is False


# -------------------------------------------------------------
# Unit Tests for SentenceTransformerProvider Lazy Loading
# -------------------------------------------------------------

def test_sentence_transformer_lazy_loading(monkeypatch):
    """Verifies that the provider does NOT instantiate model weights upon import/creation."""
    provider = SentenceTransformerProvider(config=DEFAULT_EMBEDDING_CONFIG)
    assert provider._model is None, "Model must not be loaded at instantiation time"


# -------------------------------------------------------------
# Integration Tests for SQLite Embedding Persistence
# -------------------------------------------------------------

def test_embedding_repository_save_and_retrieve(tmp_path):
    from pathlib import Path
    from app.models.document import Document, DocumentType
    from app.repositories.document_repository import DocumentRepository

    db_file = tmp_path / "embed_test.db"
    doc_repo = DocumentRepository(db_path=Path(db_file))
    doc_repo.create(
        Document(
            id="doc_persist_1",
            original_filename="test.txt",
            stored_filename="doc_persist_1_test.txt",
            name="test.txt",
            file_type=DocumentType.TXT,
            file_size=100,
            size="100 B",
            upload_timestamp="2026-11-10T00:00:00",
            hash="sha256:123",
            extracted_text="Sample text",
        )
    )
    repo = EmbeddingRepository(db_path=Path(db_file))

    dummy_vector = [0.05 * (i % 10) for i in range(384)]
    embeddings = [
        ChunkEmbedding(
            document_id="doc_persist_1",
            chunk_id=f"chk_doc_persist_1_{i:04d}",
            chunk_index=i,
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            embedding_dimension=384,
            vector=dummy_vector,
        )
        for i in range(3)
    ]

    saved = repo.save_embeddings("doc_persist_1", embeddings)
    assert len(saved) == 3
    assert repo.get_count("doc_persist_1") == 3
    assert repo.get_model_name_for_document("doc_persist_1") == "sentence-transformers/all-MiniLM-L6-v2"

    # Retrieve with vectors
    retrieved = repo.get_embeddings_by_document("doc_persist_1", include_vectors=True)
    assert len(retrieved) == 3
    assert retrieved[0].chunk_id == "chk_doc_persist_1_0000"
    assert len(retrieved[0].vector) == 384
    assert pytest.approx(retrieved[0].vector[0]) == dummy_vector[0]

    # Retrieve without vectors (metadata summary)
    summary_list = repo.get_embeddings_by_document("doc_persist_1", include_vectors=False)
    assert len(summary_list) == 3
    assert summary_list[0].vector == []


# -------------------------------------------------------------
# API Integration Tests for /api/documents/{id}/embeddings
# -------------------------------------------------------------

def test_api_generate_and_get_embeddings(client):
    # 1. Upload sample document
    content = b"Paragraph 1 regarding vector search.\n\nParagraph 2 regarding semantic chunking."
    files = {"file": ("embed_api_doc.txt", io.BytesIO(content), "text/plain")}
    upload_res = client.post("/api/documents/upload", files=files, data={"tag": "Technical"})
    assert upload_res.status_code == 201
    doc_id = upload_res.json()["id"]

    # 2. POST /api/documents/{id}/embeddings
    post_res = client.post(f"/api/documents/{doc_id}/embeddings")
    assert post_res.status_code == 200
    data = post_res.json()
    assert data["document_id"] == doc_id
    assert data["model_name"] == "sentence-transformers/all-MiniLM-L6-v2"
    assert data["embedding_dimension"] == 384
    assert data["chunks_processed"] >= 1
    assert data["embeddings_created"] == data["chunks_processed"]
    assert data["status"] == "completed"

    # 3. GET /api/documents/{id}/embeddings
    get_res = client.get(f"/api/documents/{doc_id}/embeddings")
    assert get_res.status_code == 200
    summary = get_res.json()
    assert summary["document_id"] == doc_id
    assert summary["model_name"] == "sentence-transformers/all-MiniLM-L6-v2"
    assert summary["embedding_dimension"] == 384
    assert summary["total_embeddings"] == data["embeddings_created"]
    assert summary["status"] == "completed"
    assert summary["normalized"] is True
    # Ensure vectors are not exposed in summary response
    assert "vector" not in summary


def test_api_embeddings_idempotent_regeneration(client):
    files = {"file": ("idempotent_embed.txt", io.BytesIO(b"Sentence one.\n\nSentence two."), "text/plain")}
    upload_res = client.post("/api/documents/upload", files=files)
    doc_id = upload_res.json()["id"]

    # Call 1
    res1 = client.post(f"/api/documents/{doc_id}/embeddings")
    assert res1.status_code == 200
    count1 = res1.json()["embeddings_created"]

    # Call 2 (re-uses existing without duplicate rows)
    res2 = client.post(f"/api/documents/{doc_id}/embeddings")
    assert res2.status_code == 200
    count2 = res2.json()["embeddings_created"]
    assert count1 == count2

    # GET verification
    get_res = client.get(f"/api/documents/{doc_id}/embeddings")
    assert get_res.json()["total_embeddings"] == count1


def test_api_embeddings_missing_document(client):
    missing_id = "doc-nonexistent-777777777777"
    post_res = client.post(f"/api/documents/{missing_id}/embeddings")
    assert post_res.status_code == 404
    assert post_res.json()["error"] is True

    get_res = client.get(f"/api/documents/{missing_id}/embeddings")
    assert get_res.status_code == 404
    assert get_res.json()["error"] is True


def test_model_version_mismatch_triggers_recomputation():
    """Verifies that when a different model is requested, existing embeddings are replaced."""
    from pathlib import Path
    import tempfile
    temp_dir = tempfile.mkdtemp()
    db_path = Path(temp_dir) / "mismatch_test.db"

    repo = EmbeddingRepository(db_path=db_path)
    doc_id = "doc_mismatch_test"

    from app.models.document import Document, DocumentType
    from app.repositories.document_repository import DocumentRepository
    doc_repo = DocumentRepository(db_path=db_path)
    doc_repo.create(
        Document(
            id=doc_id,
            original_filename="mismatch.txt",
            stored_filename="mismatch.txt",
            name="mismatch.txt",
            file_type=DocumentType.TXT,
            file_size=100,
            size="100 B",
            upload_timestamp="2026-11-10T00:00:00",
            hash="sha256:123",
            extracted_text="Sample text",
        )
    )

    # Save initial embeddings with Model A
    vec_a = [0.1] * 384
    repo.save_embeddings(
        doc_id,
        [ChunkEmbedding(document_id=doc_id, chunk_id=f"chk_{doc_id}_0000", chunk_index=0, model_name="model-v1", vector=vec_a)],
    )
    assert repo.get_model_name_for_document(doc_id) == "model-v1"

    # Save new embeddings with Model B (replaces model-v1)
    vec_b = [0.2] * 384
    repo.save_embeddings(
        doc_id,
        [ChunkEmbedding(document_id=doc_id, chunk_id=f"chk_{doc_id}_0000", chunk_index=0, model_name="model-v2", vector=vec_b)],
    )
    assert repo.get_model_name_for_document(doc_id) == "model-v2"
    retrieved = repo.get_embeddings_by_document(doc_id, include_vectors=True)
    assert retrieved[0].model_name == "model-v2"
    assert pytest.approx(retrieved[0].vector[0]) == 0.2


# -------------------------------------------------------------
# Optional Integration Test for Real SentenceTransformer Model
# -------------------------------------------------------------

@pytest.mark.skipif(
    os.environ.get("RUN_REAL_MODEL_TESTS") != "1",
    reason="Skipped to avoid downloading HuggingFace model in standard test runs. Set RUN_REAL_MODEL_TESTS=1 to enable.",
)
def test_real_sentence_transformer_model():
    provider = SentenceTransformerProvider(
        config=EmbeddingConfig(model_name="sentence-transformers/all-MiniLM-L6-v2", device="cpu")
    )
    vector = provider.embed_text("This is an integration test with real HuggingFace weights.")
    assert len(vector) == 384
    assert provider.validate_vector(vector) is True
