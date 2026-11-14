import math
import shutil
import tempfile
from pathlib import Path
import pytest
from app.core.errors import DimensionMismatchError, IndexingError
from app.rag.retrieval.vector_store import ChromaVectorStore


@pytest.fixture
def temp_chroma_store():
    tmp_dir = tempfile.mkdtemp(prefix="test_chroma_")
    store = ChromaVectorStore(
        persist_directory=Path(tmp_dir),
        collection_name="test_collection",
        expected_dimension=384,
    )
    yield store
    shutil.rmtree(tmp_dir, ignore_errors=True)


def test_persistent_collection_creation(temp_chroma_store):
    col = temp_chroma_store.collection
    assert col.name == "test_collection"
    assert col.metadata.get("hnsw:space") == "cosine"
    assert temp_chroma_store.count() == 0


def test_vector_insertion_and_retrieval(temp_chroma_store):
    doc_id = "doc-test-01"
    chunk_ids = ["chk_01_00", "chk_01_01"]
    # 384-dimensional normalized vector
    vec1 = [1.0 / math.sqrt(384)] * 384
    vec2 = [-1.0 / math.sqrt(384)] * 384
    embeddings = [vec1, vec2]
    documents = ["Primary database indexing passage", "Secondary caching strategy"]
    metadatas = [
        {"document_id": doc_id, "chunk_index": 0, "source_filename": "spec.txt", "file_type": "txt", "page_start": None, "page_end": None, "embedding_model": "mock-model"},
        {"document_id": doc_id, "chunk_index": 1, "source_filename": "spec.txt", "file_type": "txt", "page_start": None, "page_end": None, "embedding_model": "mock-model"},
    ]

    count = temp_chroma_store.upsert_chunks(
        document_id=doc_id,
        chunk_ids=chunk_ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas,
    )
    assert count == 2
    assert temp_chroma_store.count() == 2
    assert temp_chroma_store.get_document_chunk_count(doc_id) == 2
    assert temp_chroma_store.is_document_indexed(doc_id) is True

    # Query with identical vector to vec1
    query_results = temp_chroma_store.query(query_embedding=vec1, top_k=2)
    assert len(query_results) == 2
    top = query_results[0]
    assert top["chunk_id"] == "chk_01_00"
    assert top["content"] == "Primary database indexing passage"
    # Cosine distance between identical normalized vectors should be ~0.0
    assert pytest.approx(top["distance"], abs=1e-4) == 0.0
    assert pytest.approx(top["similarity_score"], abs=1e-4) == 1.0


def test_correct_metadata_association(temp_chroma_store):
    doc_id = "doc-pdf-02"
    chunk_ids = ["chk_02_00"]
    vec = [0.05] * 384
    norm = math.sqrt(sum(x * x for x in vec))
    normalized_vec = [x / norm for x in vec]

    metadatas = [{
        "document_id": doc_id,
        "chunk_id": "chk_02_00",
        "chunk_index": 0,
        "source_filename": "manual.pdf",
        "file_type": "pdf",
        "page_start": 4,
        "page_end": 5,
        "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
    }]

    temp_chroma_store.upsert_chunks(
        document_id=doc_id,
        chunk_ids=chunk_ids,
        embeddings=[normalized_vec],
        documents=["Page 4 manual procedures"],
        metadatas=metadatas,
    )

    results = temp_chroma_store.query(query_embedding=normalized_vec, top_k=1)
    assert len(results) == 1
    res = results[0]
    assert res["document_id"] == doc_id
    assert res["chunk_id"] == "chk_02_00"
    assert res["chunk_index"] == 0
    assert res["source_filename"] == "manual.pdf"
    assert res["file_type"] == "pdf"
    assert res["page_start"] == 4
    assert res["page_end"] == 5
    assert res["embedding_model"] == "sentence-transformers/all-MiniLM-L6-v2"


def test_cosine_distance_configuration(temp_chroma_store):
    assert temp_chroma_store.collection.metadata.get("hnsw:space") == "cosine"


def test_dimension_mismatch(temp_chroma_store):
    # Vector length 10 instead of 384
    invalid_vec = [0.1] * 10
    with pytest.raises(DimensionMismatchError) as exc_info:
        temp_chroma_store.upsert_chunks(
            document_id="doc-err",
            chunk_ids=["chk_err"],
            embeddings=[invalid_vec],
            documents=["invalid text"],
            metadatas=[{"document_id": "doc-err"}],
        )
    assert exc_info.value.status_code == 400

    # Query with invalid dimension
    with pytest.raises(DimensionMismatchError):
        temp_chroma_store.query(query_embedding=[0.1] * 10, top_k=5)


def test_safe_reindexing_and_idempotency(temp_chroma_store):
    doc_id = "doc-reindex"
    vec = [1.0 / math.sqrt(384)] * 384

    # Initial indexing: 3 chunks
    initial_chunks = ["chk_r_00", "chk_r_01", "chk_r_02"]
    temp_chroma_store.upsert_chunks(
        document_id=doc_id,
        chunk_ids=initial_chunks,
        embeddings=[vec, vec, vec],
        documents=["Text A", "Text B", "Text C"],
        metadatas=[{"document_id": doc_id, "chunk_index": i} for i in range(3)],
    )
    assert temp_chroma_store.count() == 3

    # Re-indexing with only 1 chunk (e.g. chunk size increased)
    temp_chroma_store.upsert_chunks(
        document_id=doc_id,
        chunk_ids=["chk_r_new_00"],
        embeddings=[vec],
        documents=["Text Combined"],
        metadatas=[{"document_id": doc_id, "chunk_index": 0}],
    )
    # Stale chunks must be cleaned up; exactly 1 chunk remains
    assert temp_chroma_store.count() == 1
    assert temp_chroma_store.get_document_chunk_count(doc_id) == 1


def test_restart_persistence():
    tmp_dir = tempfile.mkdtemp(prefix="test_persist_")
    vec = [1.0 / math.sqrt(384)] * 384
    try:
        # Phase 1: create store and index
        store1 = ChromaVectorStore(
            persist_directory=Path(tmp_dir),
            collection_name="persist_col",
            expected_dimension=384,
        )
        store1.upsert_chunks(
            document_id="doc-persist",
            chunk_ids=["chk-persist-01"],
            embeddings=[vec],
            documents=["Persisted text across process restart"],
            metadatas=[{"document_id": "doc-persist", "chunk_index": 0}],
        )
        assert store1.count() == 1
        del store1

        # Phase 2: simulate restart by creating fresh ChromaVectorStore pointing to same directory
        store2 = ChromaVectorStore(
            persist_directory=Path(tmp_dir),
            collection_name="persist_col",
            expected_dimension=384,
        )
        assert store2.count() == 1
        results = store2.query(query_embedding=vec, top_k=1)
        assert len(results) == 1
        assert results[0]["content"] == "Persisted text across process restart"
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
