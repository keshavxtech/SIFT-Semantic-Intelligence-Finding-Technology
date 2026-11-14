import io
import os
from pathlib import Path
import pymupdf
import pytest
from app.core.config import settings


def create_sample_pdf_bytes(text: str = "SIFT Document: Evaluation of dense retrieval benchmarks.") -> bytes:
    """Generates a valid single-page PDF with selectable text."""
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((50, 72), text)
    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


def create_blank_pdf_bytes() -> bytes:
    """Generates a valid PDF with zero text."""
    doc = pymupdf.open()
    doc.new_page()  # Blank page without text
    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


def test_valid_pdf_upload(client):
    pdf_bytes = create_sample_pdf_bytes("Enterprise Security Architecture Specifications")
    files = {"file": ("security_arch.pdf", io.BytesIO(pdf_bytes), "application/pdf")}
    data = {"tag": "Security", "description": "Enterprise Security Specs"}

    response = client.post("/api/documents/upload", files=files, data=data)
    assert response.status_code == 201
    result = response.json()
    assert result["name"] == "security_arch.pdf"
    assert result["file_type"] == "pdf"
    assert result["type"] == "pdf"
    assert result["processing_status"] == "indexed"
    assert result["status"] == "indexed"
    assert result["page_count"] == 1
    assert result["tag"] == "Security"
    assert "sha256:" in result["hash"]
    assert "Enterprise Security Architecture" in result["raw_text_preview"]
    assert "file_path" not in result  # Path secrecy requirement


def test_valid_txt_upload(client):
    text_content = (
        "SIFT Technical Overview.\n"
        "Section 1: Data ingestion pipeline with strict security controls.\n"
        "Section 2: Deterministic text extraction."
    ).encode("utf-8")
    files = {"file": ("tech_overview.txt", io.BytesIO(text_content), "text/plain")}
    data = {"tag": "Technical", "description": "System architecture overview"}

    response = client.post("/api/documents/upload", files=files, data=data)
    assert response.status_code == 201
    result = response.json()
    assert result["name"] == "tech_overview.txt"
    assert result["file_type"] == "txt"
    assert result["processing_status"] == "indexed"
    assert result["page_count"] >= 1
    assert "SIFT Technical Overview" in result["raw_text_preview"]


def test_unsupported_file_extension(client):
    files = {"file": ("malware.exe", io.BytesIO(b"MZ\x90\x00\x03\x00\x00\x00"), "application/x-dosexec")}
    response = client.post("/api/documents/upload", files=files)
    assert response.status_code == 400
    data = response.json()
    assert data["error"] is True
    assert "not supported" in data["message"].lower()


def test_empty_file_upload(client):
    files = {"file": ("empty_file.txt", io.BytesIO(b""), "text/plain")}
    response = client.post("/api/documents/upload", files=files)
    assert response.status_code == 400
    data = response.json()
    assert data["error"] is True
    assert "empty" in data["message"].lower()


def test_corrupted_pdf_upload(client):
    corrupted_bytes = b"%PDF-1.5 fake corrupted content that cannot be opened as pdf"
    files = {"file": ("corrupted.pdf", io.BytesIO(corrupted_bytes), "application/pdf")}
    response = client.post("/api/documents/upload", files=files)
    assert response.status_code == 422
    data = response.json()
    assert data["error"] is True
    assert "corrupted" in data["message"].lower()


def test_pdf_with_no_extractable_text(client):
    blank_pdf = create_blank_pdf_bytes()
    files = {"file": ("blank_canvas.pdf", io.BytesIO(blank_pdf), "application/pdf")}
    response = client.post("/api/documents/upload", files=files)
    assert response.status_code == 422
    data = response.json()
    assert data["error"] is True
    assert "no extractable" in data["message"].lower()


def test_oversized_file_upload(client):
    # Default limit is 10 MB; send 11 MB of plain bytes
    oversized_bytes = b"0" * (11 * 1024 * 1024)
    files = {"file": ("oversized.txt", io.BytesIO(oversized_bytes), "text/plain")}
    response = client.post("/api/documents/upload", files=files)
    assert response.status_code == 413
    data = response.json()
    assert data["error"] is True
    assert "exceeds maximum allowed limit" in data["message"].lower()


def test_duplicate_filenames_no_overwrite(client):
    # Upload first document
    content1 = b"Version 1 of duplicate file test content."
    files1 = {"file": ("report.txt", io.BytesIO(content1), "text/plain")}
    res1 = client.post("/api/documents/upload", files=files1)
    assert res1.status_code == 201
    doc1 = res1.json()

    # Upload second document with exact same filename but different text
    content2 = b"Version 2 of duplicate file test content."
    files2 = {"file": ("report.txt", io.BytesIO(content2), "text/plain")}
    res2 = client.post("/api/documents/upload", files=files2)
    assert res2.status_code == 201
    doc2 = res2.json()

    # Must be assigned distinct IDs and stored without collision
    assert doc1["id"] != doc2["id"]
    assert doc1["stored_filename"] != doc2["stored_filename"]

    # Verify both documents exist independently and accurately in the system
    detail1 = client.get(f"/api/documents/{doc1['id']}").json()
    detail2 = client.get(f"/api/documents/{doc2['id']}").json()
    assert "Version 1" in detail1["extracted_text"]
    assert "Version 2" in detail2["extracted_text"]


def test_path_traversal_sanitization(client):
    content = b"Attempted path traversal upload."
    files = {"file": ("../../../../etc/passwd.txt", io.BytesIO(content), "text/plain")}
    res = client.post("/api/documents/upload", files=files)
    assert res.status_code == 201
    doc = res.json()
    # Name must be stripped of traversal slashes
    assert "/" not in doc["name"]
    assert "\\" not in doc["name"]
    assert ".." not in doc["name"]
    assert "passwd.txt" in doc["name"]


def test_txt_encoding_fallback(client):
    # Latin-1 encoded characters (e.g., 'Café résumé')
    latin1_bytes = "Rapport Financier: Coût total de l'opération en été.".encode("latin-1")
    files = {"file": ("rapport_latin1.txt", io.BytesIO(latin1_bytes), "text/plain")}
    res = client.post("/api/documents/upload", files=files)
    assert res.status_code == 201
    doc = res.json()
    assert doc["processing_status"] == "indexed"


def test_document_listing_and_detail_retrieval(client):
    # List documents
    list_res = client.get("/api/documents")
    assert list_res.status_code == 200
    data = list_res.json()
    assert "total" in data
    assert "documents" in data
    assert data["total"] >= 1

    first_doc = data["documents"][0]
    doc_id = first_doc["id"]

    # Detail retrieval includes extracted_text
    detail_res = client.get(f"/api/documents/{doc_id}")
    assert detail_res.status_code == 200
    detail = detail_res.json()
    assert detail["id"] == doc_id
    assert "extracted_text" in detail
    assert detail["extracted_text"] is not None


def test_document_deletion_removes_db_and_disk_file(client):
    # Upload a file specifically for deletion
    content = b"Document scheduled for immediate permanent deletion."
    files = {"file": ("to_delete.txt", io.BytesIO(content), "text/plain")}
    res = client.post("/api/documents/upload", files=files)
    assert res.status_code == 201
    doc = res.json()
    doc_id = doc["id"]
    stored_filename = doc["stored_filename"]

    # Physical file should exist in storage directory
    storage_path = Path(settings.STORAGE_DIR) / stored_filename
    assert storage_path.exists()

    # Issue DELETE request
    del_res = client.delete(f"/api/documents/{doc_id}")
    assert del_res.status_code == 204

    # DB lookup should now 404
    lookup_res = client.get(f"/api/documents/{doc_id}")
    assert lookup_res.status_code == 404

    # Physical file on disk must be removed
    assert not storage_path.exists()


def test_missing_document_id_handling(client):
    missing_id = "doc-nonexistent-000000000000"

    # GET missing ID returns 404
    get_res = client.get(f"/api/documents/{missing_id}")
    assert get_res.status_code == 404
    assert get_res.json()["error"] is True

    # DELETE missing ID returns 404
    del_res = client.delete(f"/api/documents/{missing_id}")
    assert del_res.status_code == 404
    assert del_res.json()["error"] is True
