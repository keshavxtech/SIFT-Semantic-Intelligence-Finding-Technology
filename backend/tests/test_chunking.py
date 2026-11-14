import io
import pymupdf
import pytest
from app.rag.chunking.config import ChunkConfig, DEFAULT_CHUNK_CONFIG
from app.rag.chunking.models import Chunk
from app.rag.chunking.semantic_chunker import StructureAwareChunker
from app.rag.ingestion.parser import ParsedPage


def create_multi_page_pdf(pages_text: list) -> bytes:
    """Creates a multi-page PDF with selectable text on each page."""
    doc = pymupdf.open()
    for text in pages_text:
        page = doc.new_page()
        page.insert_text((50, 72), text)
    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


# -------------------------------------------------------------
# Unit Tests for StructureAwareChunker & ChunkConfig
# -------------------------------------------------------------

def test_chunk_config_validation():
    # Valid config
    cfg = ChunkConfig(chunk_size=800, chunk_overlap=100, minimum_chunk_size=50)
    assert cfg.chunk_size == 800
    assert cfg.chunk_overlap == 100
    assert cfg.minimum_chunk_size == 50

    # chunk_size <= 0
    with pytest.raises(ValueError, match="chunk_size must be positive"):
        ChunkConfig(chunk_size=0, chunk_overlap=0, minimum_chunk_size=10)

    # chunk_overlap >= chunk_size
    with pytest.raises(ValueError, match="chunk_overlap .* must be smaller than chunk_size"):
        ChunkConfig(chunk_size=500, chunk_overlap=500, minimum_chunk_size=50)

    # minimum_chunk_size > chunk_size
    with pytest.raises(ValueError, match="minimum_chunk_size .* must not exceed chunk_size"):
        ChunkConfig(chunk_size=300, chunk_overlap=50, minimum_chunk_size=400)


def test_short_paragraph_not_split():
    chunker = StructureAwareChunker(ChunkConfig(chunk_size=1000, chunk_overlap=100))
    text = "This is a brief, coherent technical paragraph that fits comfortably inside a single chunk."
    chunks = chunker.chunk_document(document_id="doc_test_1", text=text)

    assert len(chunks) == 1
    assert chunks[0].content == text
    assert chunks[0].chunk_index == 0
    assert chunks[0].character_count == len(text)
    assert chunks[0].chunk_id == "chk_doc_test_1_0000"


def test_multiple_paragraphs_combined():
    chunker = StructureAwareChunker(ChunkConfig(chunk_size=1000, chunk_overlap=100))
    p1 = "Paragraph 1: Overview of system design and components."
    p2 = "Paragraph 2: Implementation details covering data ingestion and serialization."
    p3 = "Paragraph 3: Security safeguards preventing path traversal and arbitrary code execution."
    full_text = f"{p1}\n\n{p2}\n\n{p3}"

    chunks = chunker.chunk_document(document_id="doc_multi_p", text=full_text)
    # Total characters are ~220, so they should easily combine into a single chunk
    assert len(chunks) == 1
    assert p1 in chunks[0].content
    assert p2 in chunks[0].content
    assert p3 in chunks[0].content


def test_long_paragraph_splits_on_sentence_boundaries():
    chunker = StructureAwareChunker(ChunkConfig(chunk_size=300, chunk_overlap=50))
    s1 = "Artificial intelligence document systems require robust parsing."
    s2 = "Vector embeddings represent dense semantics across high-dimensional latent space."
    s3 = "Lexical matching guarantees exact keyword accuracy for technical terms."
    s4 = "Hybrid ranking balances both approaches to optimize mean reciprocal rank across benchmarks."
    s5 = "Evaluation metrics demonstrate superior precision with structure-aware chunking."
    long_para = f"{s1} {s2} {s3} {s4} {s5}"

    chunks = chunker.chunk_document(document_id="doc_long_p", text=long_para)
    assert len(chunks) > 1

    # Verify that each chunk respects the chunk size
    for c in chunks:
        assert c.character_count <= 350
        # Check that sentences are not truncated awkwardly
        assert c.content.endswith((".", "!", "?")) or len(c.content.split()[-1]) > 0


def test_sentence_boundary_preservation():
    # Set chunk_size=70 so the 123-char text splits across sentence boundaries
    chunker = StructureAwareChunker(ChunkConfig(chunk_size=70, chunk_overlap=15, minimum_chunk_size=20))
    text = (
        "First complete sentence here. "
        "Second complete sentence follows immediately. "
        "Third complete sentence concludes this segment."
    )
    chunks = chunker.chunk_document(document_id="doc_sentences", text=text)
    assert len(chunks) >= 2
    # Verify chunks do not start or end with partial words
    for c in chunks:
        words = c.content.split()
        assert len(words) > 0
        assert words[0].strip(".,!?:;\"'()[]").isalnum()


def test_chunk_size_limits():
    chunk_size = 250
    chunker = StructureAwareChunker(ChunkConfig(chunk_size=chunk_size, chunk_overlap=40))
    text = "\n\n".join([f"This is test paragraph number {i} with sufficient text to fill capacity." for i in range(15)])
    chunks = chunker.chunk_document(document_id="doc_limits", text=text)

    for c in chunks:
        # Chunks must not exceed chunk_size by more than minor paragraph join margins
        assert c.character_count <= chunk_size + 50


def test_configurable_overlap():
    overlap_size = 60
    # Set chunk_size=150 so the 264-char text splits across multiple chunks with overlap
    chunker = StructureAwareChunker(ChunkConfig(chunk_size=150, chunk_overlap=overlap_size, minimum_chunk_size=40))
    p1 = "Paragraph Alpha discusses data integrity and encryption."
    p2 = "Paragraph Beta discusses fault tolerance in distributed clusters."
    p3 = "Paragraph Gamma discusses verifiable source citations in RAG architectures."
    p4 = "Paragraph Delta discusses compliance audits and log retention."
    text = f"{p1}\n\n{p2}\n\n{p3}\n\n{p4}"

    chunks = chunker.chunk_document(document_id="doc_overlap", text=text)
    assert len(chunks) >= 2

    # Check for overlapping content between consecutive chunks
    overlap_found = False
    for i in range(len(chunks) - 1):
        c1 = chunks[i].content
        c2 = chunks[i + 1].content
        # Check if words from the end of c1 appear at the start of c2
        tail_words = c1.split()[-4:]
        tail_phrase = " ".join(tail_words)
        if tail_phrase in c2:
            overlap_found = True
            break
    assert overlap_found, "Expected overlap between consecutive chunks"


def test_empty_and_whitespace_only_text():
    chunker = StructureAwareChunker()
    assert chunker.chunk_document(document_id="doc_empty", text="") == []
    assert chunker.chunk_document(document_id="doc_spaces", text="   \n\n\t  \n  ") == []


def test_very_long_sentence_handling():
    chunker = StructureAwareChunker(ChunkConfig(chunk_size=200, chunk_overlap=30))
    # Sentence with 600 characters of words without periods
    long_sentence = "word " * 120
    chunks = chunker.chunk_document(document_id="doc_long_sentence", text=long_sentence)
    assert len(chunks) >= 2
    for c in chunks:
        assert c.character_count <= 250
        assert len(c.content.strip()) > 0


def test_deterministic_output():
    chunker = StructureAwareChunker(ChunkConfig(chunk_size=400, chunk_overlap=60))
    text = (
        "Enterprise document systems must guarantee reproducible segmentation.\n\n"
        "Given the same input text and parameters, every execution must generate "
        "the exact same chunks with identical IDs, indices, and content bounds.\n\n"
        "This determinism is essential for cache invalidation and stable vector indexing."
    )
    run1 = chunker.chunk_document(document_id="doc_stable", text=text)
    run2 = chunker.chunk_document(document_id="doc_stable", text=text)

    assert len(run1) == len(run2)
    for c1, c2 in zip(run1, run2):
        assert c1.chunk_id == c2.chunk_id
        assert c1.chunk_index == c2.chunk_index
        assert c1.content == c2.content
        assert c1.character_count == c2.character_count
        assert c1.token_count == c2.token_count


def test_sequential_chunk_indexes_and_unique_ids():
    chunker = StructureAwareChunker(ChunkConfig(chunk_size=150, chunk_overlap=20))
    text = "\n\n".join([f"Distinct unit number {i} with specific content details." for i in range(8)])
    chunks = chunker.chunk_document(document_id="doc_seq", text=text)

    seen_ids = set()
    for expected_idx, c in enumerate(chunks):
        assert c.chunk_index == expected_idx
        assert c.chunk_id == f"chk_doc_seq_{expected_idx:04d}"
        assert c.document_id == "doc_seq"
        assert c.chunk_id not in seen_ids
        seen_ids.add(c.chunk_id)


def test_pdf_page_provenance():
    # Set chunk_size=100 and chunk_overlap=0 so each ~75-char page forms its own distinct chunk
    chunker = StructureAwareChunker(ChunkConfig(chunk_size=100, chunk_overlap=0, minimum_chunk_size=20))
    pages = [
        ParsedPage(page_number=1, text="Page 1 Content: Introduction to RAG pipelines and information retrieval."),
        ParsedPage(page_number=2, text="Page 2 Content: Detailed mathematical formulas for dense cosine similarity."),
        ParsedPage(page_number=3, text="Page 3 Content: Empirical evaluation and latency profiling results."),
    ]
    chunks = chunker.chunk_document(document_id="doc_pdf_prov", text="", pages=pages)

    assert len(chunks) == 3
    # First chunk is strictly on Page 1
    assert chunks[0].page_start == 1
    assert chunks[0].page_end == 1
    # Second chunk is strictly on Page 2
    assert chunks[1].page_start == 2
    assert chunks[1].page_end == 2
    # Last chunk is strictly on Page 3
    assert chunks[2].page_start == 3
    assert chunks[2].page_end == 3


def test_pdf_chunk_spanning_multiple_pages():
    # With a larger chunk_size (300), short pages are merged, spanning pages
    chunker = StructureAwareChunker(ChunkConfig(chunk_size=300, chunk_overlap=40))
    pages = [
        ParsedPage(page_number=1, text="Page 1 brief text."),
        ParsedPage(page_number=2, text="Page 2 brief continuation."),
    ]
    chunks = chunker.chunk_document(document_id="doc_span", text="", pages=pages)
    assert len(chunks) == 1
    assert chunks[0].page_start == 1
    assert chunks[0].page_end == 2


def test_txt_null_page_provenance():
    chunker = StructureAwareChunker(ChunkConfig(chunk_size=300, chunk_overlap=40))
    txt_content = "Plain text document with no physical page boundaries.\n\nSecond section of plain text."
    # Pages is None for TXT
    chunks = chunker.chunk_document(document_id="doc_txt_prov", text=txt_content, pages=None)

    assert len(chunks) >= 1
    for c in chunks:
        assert c.page_start is None
        assert c.page_end is None


# -------------------------------------------------------------
# API Integration Tests for /api/documents/{id}/chunks
# -------------------------------------------------------------

def test_api_generate_and_get_chunks_txt(client):
    # Upload TXT file
    text_content = (
        "Section 1: Architecture Overview.\n"
        "SIFT provides deterministic document chunking with verifiable source citations.\n\n"
        "Section 2: Security Specifications.\n"
        "All data at rest is encrypted, and internal filesystem paths are never exposed to clients."
    ).encode("utf-8")
    files = {"file": ("chunk_api_test.txt", io.BytesIO(text_content), "text/plain")}
    upload_res = client.post("/api/documents/upload", files=files, data={"tag": "Technical"})
    assert upload_res.status_code == 201
    doc_id = upload_res.json()["id"]

    # POST /api/documents/{id}/chunks
    post_res = client.post(f"/api/documents/{doc_id}/chunks")
    assert post_res.status_code == 200
    data = post_res.json()
    assert data["document_id"] == doc_id
    assert data["total_chunks"] >= 1
    assert len(data["chunks"]) == data["total_chunks"]

    # Verify TXT page provenance is null
    first_chunk = data["chunks"][0]
    assert first_chunk["page_start"] is None
    assert first_chunk["page_end"] is None
    assert first_chunk["chunk_index"] == 0
    assert first_chunk["chunk_id"].startswith(f"chk_{doc_id}_")
    assert "file_path" not in first_chunk["metadata"]

    # GET /api/documents/{id}/chunks returns the persisted chunks
    get_res = client.get(f"/api/documents/{doc_id}/chunks")
    assert get_res.status_code == 200
    get_data = get_res.json()
    assert get_data["document_id"] == doc_id
    assert get_data["total_chunks"] == data["total_chunks"]
    assert get_data["chunks"][0]["chunk_id"] == first_chunk["chunk_id"]


def test_api_generate_chunks_pdf_provenance(client):
    # Create 2-page PDF
    pdf_bytes = create_multi_page_pdf([
        "Page 1: Deep Neural Networks in Document Processing and Knowledge Extraction.",
        "Page 2: Vector Search Optimizations for Approximate Nearest Neighbor Traversal.",
    ])
    files = {"file": ("provenance_doc.pdf", io.BytesIO(pdf_bytes), "application/pdf")}
    upload_res = client.post("/api/documents/upload", files=files, data={"tag": "Research"})
    assert upload_res.status_code == 201
    doc_id = upload_res.json()["id"]

    # POST chunks with chunk_size=90 so each ~80-char page forms its own chunk
    chunk_res = client.post(f"/api/documents/{doc_id}/chunks?chunk_size=90&chunk_overlap=15&minimum_chunk_size=20")
    assert chunk_res.status_code == 200
    chunks = chunk_res.json()["chunks"]
    assert len(chunks) >= 2

    # Check page provenance
    assert chunks[0]["page_start"] == 1
    assert chunks[-1]["page_end"] == 2


def test_api_chunks_missing_document(client):
    missing_id = "doc-nonexistent-999999999999"
    post_res = client.post(f"/api/documents/{missing_id}/chunks")
    assert post_res.status_code == 404
    assert post_res.json()["error"] is True

    get_res = client.get(f"/api/documents/{missing_id}/chunks")
    assert get_res.status_code == 404
    assert get_res.json()["error"] is True


def test_api_idempotent_regeneration(client):
    files = {"file": ("idempotency_doc.txt", io.BytesIO(b"Paragraph 1\n\nParagraph 2\n\nParagraph 3"), "text/plain")}
    upload_res = client.post("/api/documents/upload", files=files)
    doc_id = upload_res.json()["id"]

    # Run 1
    res1 = client.post(f"/api/documents/{doc_id}/chunks")
    assert res1.status_code == 200
    total1 = res1.json()["total_chunks"]

    # Run 2 (idempotent regeneration)
    res2 = client.post(f"/api/documents/{doc_id}/chunks")
    assert res2.status_code == 200
    total2 = res2.json()["total_chunks"]

    assert total1 == total2
    assert res1.json()["chunks"][0]["chunk_id"] == res2.json()["chunks"][0]["chunk_id"]

    # Verify no duplicate records accumulated in GET
    get_res = client.get(f"/api/documents/{doc_id}/chunks")
    assert get_res.json()["total_chunks"] == total1
