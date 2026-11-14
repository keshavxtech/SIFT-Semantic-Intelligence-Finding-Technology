import io
import pytest
from app.core.errors import DocumentNotFoundError, IndexingError, ModelMismatchError
from app.models.document import Document, DocumentType, ProcessingStatus
from app.rag.chunking.models import Chunk
from app.rag.embeddings.models import ChunkEmbedding
from app.services.indexing_service import indexing_service


def test_index_missing_document_404(client):
    res = client.post("/api/documents/doc-non-existent-999/index")
    assert res.status_code == 404

    status_res = client.get("/api/documents/doc-non-existent-999/index-status")
    assert status_res.status_code == 404


def test_index_document_without_chunks_400(client):
    # Upload text without generating chunks
    files = {"file": ("empty_chunks.txt", io.BytesIO(b"Document text without chunking."), "text/plain")}
    up = client.post("/api/documents/upload", files=files)
    doc_id = up.json()["id"]

    res = client.post(f"/api/documents/{doc_id}/index")
    assert res.status_code == 400
    assert "has no chunks" in res.json()["message"]


def test_index_document_without_embeddings_400(client):
    files = {"file": ("no_embeds.txt", io.BytesIO(b"Document text with chunks but no embeddings."), "text/plain")}
    up = client.post("/api/documents/upload", files=files)
    doc_id = up.json()["id"]

    # Generate chunks only
    client.post(f"/api/documents/{doc_id}/chunks")

    res = client.post(f"/api/documents/{doc_id}/index")
    assert res.status_code == 400
    assert "has no generated embeddings" in res.json()["message"]


def test_index_model_mismatch_triggers_error():
    doc_id = "doc-mismatch-test"
    doc = Document(
        id=doc_id,
        original_filename="mismatch.txt",
        stored_filename="mismatch.txt",
        name="Mismatch Test",
        file_type=DocumentType.TXT,
        file_size=100,
        size="100 B",
        upload_timestamp="12:00 PM",
        hash="mismatchhash",
    )
    indexing_service.doc_repo.create(doc)

    # Save 1 chunk
    chunk = Chunk(
        chunk_id=f"chk_{doc_id}_0000",
        document_id=doc_id,
        chunk_index=0,
        content="Testing model mismatch detection.",
        character_count=35,
        token_count=8,
    )
    indexing_service.chunk_repo.save_chunks(doc_id, [chunk])

    # Save embedding with old/different model name
    emb = ChunkEmbedding(
        document_id=doc_id,
        chunk_id=f"chk_{doc_id}_0000",
        chunk_index=0,
        model_name="old-incompatible-model-v1",
        vector=[0.05] * 384,
    )
    indexing_service.embed_repo.save_embeddings(doc_id, [emb])

    with pytest.raises(ModelMismatchError):
        import asyncio
        asyncio.run(indexing_service.index_document(doc_id))


def test_api_index_and_status_lifecycle(client):
    content = (
        b"Enterprise Document Architecture and Vector Retrieval.\n\n"
        b"Section 1: Vector Space Partitioning.\n"
        b"High-dimensional vector indexing utilizes HNSW graphs with cosine distance metrics."
    )
    files = {"file": ("vector_architecture.txt", io.BytesIO(content), "text/plain")}
    up = client.post("/api/documents/upload", files=files)
    assert up.status_code == 201
    doc_id = up.json()["id"]

    # Check status before indexing
    status_pre = client.get(f"/api/documents/{doc_id}/index-status")
    assert status_pre.status_code == 200
    assert status_pre.json()["status"] == "not_indexed"
    assert status_pre.json()["is_indexed"] is False

    # Generate chunks & embeddings
    client.post(f"/api/documents/{doc_id}/chunks")
    client.post(f"/api/documents/{doc_id}/embeddings")

    # Index into ChromaDB
    index_res = client.post(f"/api/documents/{doc_id}/index")
    assert index_res.status_code == 200
    data = index_res.json()
    assert data["document_id"] == doc_id
    assert data["status"] == "indexed"
    assert data["indexed_chunks"] >= 1
    assert data["embedding_dimension"] == 384

    # Check status after indexing
    status_post = client.get(f"/api/documents/{doc_id}/index-status")
    assert status_post.status_code == 200
    assert status_post.json()["status"] == "indexed"
    assert status_post.json()["is_indexed"] is True
    assert status_post.json()["indexed_chunks"] == data["indexed_chunks"]

    # Re-indexing must be idempotent
    reindex_res = client.post(f"/api/documents/{doc_id}/index")
    assert reindex_res.status_code == 200
    assert reindex_res.json()["status"] == "indexed"
    assert reindex_res.json()["indexed_chunks"] == data["indexed_chunks"]


def test_document_deletion_removes_from_vector_store(client):
    content = b"Content to be deleted from ChromaDB."
    files = {"file": ("to_delete.txt", io.BytesIO(content), "text/plain")}
    up = client.post("/api/documents/upload", files=files)
    doc_id = up.json()["id"]

    client.post(f"/api/documents/{doc_id}/chunks")
    client.post(f"/api/documents/{doc_id}/embeddings")
    client.post(f"/api/documents/{doc_id}/index")

    # Verify indexed in ChromaDB
    assert indexing_service.vector_store.is_document_indexed(doc_id) is True

    # Delete document
    del_res = client.delete(f"/api/documents/{doc_id}")
    assert del_res.status_code == 204

    # Verify vector store records are purged
    assert indexing_service.vector_store.is_document_indexed(doc_id) is False
    assert indexing_service.vector_store.get_document_chunk_count(doc_id) == 0
