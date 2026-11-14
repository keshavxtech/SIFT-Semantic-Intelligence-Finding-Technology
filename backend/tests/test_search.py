import io
import shutil
import tempfile
from pathlib import Path
import pytest
from app.rag.retrieval.retriever import SemanticRetriever
from app.rag.retrieval.vector_store import ChromaVectorStore
from app.services.search_service import search_service


def test_search_empty_query_400(client):
    res = client.post("/api/search", json={"query": "   ", "top_k": 5})
    assert res.status_code == 400


def test_search_empty_collection_returns_zero_matches(client):
    tmp = tempfile.mkdtemp(prefix="empty_search_col_")
    old_store = search_service.retriever.vector_store
    try:
        empty_store = ChromaVectorStore(persist_directory=Path(tmp), collection_name="empty_collection")
        search_service.retriever.vector_store = empty_store

        res = client.post("/api/search", json={"query": "unindexed query term", "top_k": 5})
        assert res.status_code == 200
        data = res.json()
        assert data["total_matches"] == 0
        assert data["results"] == []
    finally:
        search_service.retriever.vector_store = old_store
        shutil.rmtree(tmp, ignore_errors=True)


def test_search_non_matching_document_id_filter(client):
    res = client.post(
        "/api/search",
        json={"query": "any query", "document_ids": ["doc-non-existent-999"], "top_k": 5},
    )
    assert res.status_code == 200
    data = res.json()
    assert data["total_matches"] == 0
    assert data["results"] == []


def test_search_real_semantic_retrieval_lifecycle(client):
    # 1. Upload Doc A (Security Policy)
    content_a = (
        b"Enterprise Security Controls and Safeguards.\n\n"
        b"Section 1: Data Encryption Standards.\n"
        b"All data stores must enforce AES-256 encryption at rest and TLS 1.3 protocol in transit."
    )
    res_a = client.post("/api/documents/upload", files={"file": ("security_policy.txt", io.BytesIO(content_a), "text/plain")})
    doc_a = res_a.json()["id"]

    # 2. Upload Doc B (Financial Audit)
    content_b = (
        b"Quarterly Financial Liquidity Analysis.\n\n"
        b"Section 1: Working Capital Reserves.\n"
        b"The organization maintains sufficient cash equivalents to cover twelve months of operational overhead."
    )
    res_b = client.post("/api/documents/upload", files={"file": ("financial_audit.txt", io.BytesIO(content_b), "text/plain")})
    doc_b = res_b.json()["id"]

    # 3. Generate chunks, embeddings, and index both documents
    for doc_id in [doc_a, doc_b]:
        client.post(f"/api/documents/{doc_id}/chunks")
        client.post(f"/api/documents/{doc_id}/embeddings")
        idx_res = client.post(f"/api/documents/{doc_id}/index")
        assert idx_res.status_code == 200

    # 4. Search across all documents
    search_payload = {
        "query": "encryption at rest and TLS transit protocol",
        "top_k": 5,
    }
    response = client.post("/api/search", json=search_payload)
    assert response.status_code == 200
    data = response.json()
    assert data["query"] == "encryption at rest and TLS transit protocol"
    assert data["total_matches"] >= 1
    assert "execution_time_ms" in data
    assert "metric" in data

    # Verify top result provenance
    top = data["results"][0]
    assert "document_id" in top
    assert "chunk_id" in top
    assert "chunk_index" in top
    assert "content" in top
    assert "source_filename" in top
    assert "distance" in top
    assert "similarity_score" in top
    assert isinstance(top["distance"], float)
    assert isinstance(top["similarity_score"], float)

    # 5. Search with document_id filtering (scope strictly to Doc B)
    scoped_payload = {
        "query": "encryption at rest",
        "document_ids": [doc_b],
        "top_k": 5,
    }
    scoped_res = client.post("/api/search", json=scoped_payload)
    assert scoped_res.status_code == 200
    scoped_data = scoped_res.json()
    # Scoped to Doc B only: no Doc A chunks can appear
    for r in scoped_data["results"]:
        assert r["document_id"] == doc_b
        assert r["document_id"] != doc_a

    # 6. Verify top_k bounding
    limit_payload = {
        "query": "working capital",
        "top_k": 20,
    }
    limit_res = client.post("/api/search", json=limit_payload)
    assert limit_res.status_code == 200
    assert len(limit_res.json()["results"]) <= 20
