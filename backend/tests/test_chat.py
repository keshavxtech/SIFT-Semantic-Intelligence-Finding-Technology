import io
import pytest
from app.core.errors import (
    ChatSessionNotFoundError,
    LLMProviderError,
    LLMProviderNotConfiguredError,
    LLMRateLimitError,
    LLMTimeoutError,
)
from app.rag.generation.base import BaseLLMProvider, LLMGenerationResult
from app.rag.generation.groq_provider import GroqProvider
from app.rag.generation.mock_provider import MockLLMProvider
from app.rag.generation.prompt_builder import PromptBuilder
from app.schemas.chat import ChatRequest
from app.services.chat_service import chat_service


# -------------------------------------------------------------
# Unit Tests for LLM Provider & Prompt Builder
# -------------------------------------------------------------

@pytest.mark.asyncio
async def test_mock_llm_provider():
    provider = MockLLMProvider(simulated_answer="Test grounded response [1]")
    res = await provider.generate(
        prompt="User Question",
        system_instruction="Strict system instructions",
        conversation_history=[],
    )
    assert isinstance(res, LLMGenerationResult)
    assert res.text == "Test grounded response [1]"
    assert res.provider == "mock"
    assert res.model == "mock-rag-llama-3"


@pytest.mark.asyncio
async def test_groq_provider_missing_key_raises_error():
    provider = GroqProvider(api_key="", model="llama-3.1-8b-instant")
    with pytest.raises(LLMProviderNotConfiguredError) as exc_info:
        await provider.generate(prompt="Hello", system_instruction="System")
    assert exc_info.value.status_code == 503


@pytest.mark.asyncio
async def test_mock_provider_timeout_simulation():
    provider = MockLLMProvider(simulated_timeout=True)
    with pytest.raises(LLMTimeoutError) as exc_info:
        await provider.generate(prompt="Hello", system_instruction="System")
    assert exc_info.value.status_code == 504


@pytest.mark.asyncio
async def test_mock_provider_error_simulation():
    provider = MockLLMProvider(simulated_error="Upstream API outage")
    with pytest.raises(LLMProviderError) as exc_info:
        await provider.generate(prompt="Hello", system_instruction="System")
    assert exc_info.value.status_code == 502


def test_prompt_builder_with_passages():
    passages = [
        {
            "chunk_id": "chk-001",
            "source_filename": "policy.pdf",
            "page_start": 3,
            "page_end": 3,
            "content": "Windshield deductible is strictly AED 500.",
        },
        {
            "chunk_id": "chk-002",
            "source_filename": "terms.txt",
            "page_start": None,
            "page_end": None,
            "content": "No claims discount remains unaffected.",
        },
    ]
    prompt = PromptBuilder.build_grounded_prompt("What is the windshield deductible?", passages)
    assert "Passage [1]" in prompt
    assert "policy.pdf (Page 3)" in prompt
    assert "Windshield deductible is strictly AED 500." in prompt
    assert "Passage [2]" in prompt
    assert "Page N/A" in prompt
    assert "No claims discount remains unaffected." in prompt
    assert "USER QUESTION: What is the windshield deductible?" in prompt


def test_prompt_builder_empty_passages():
    prompt = PromptBuilder.build_grounded_prompt("What is the company valuation?", [])
    assert "NO RELEVANT DOCUMENT CONTEXT AVAILABLE" in prompt
    assert "insufficient evidence" in prompt


# -------------------------------------------------------------
# Integration Tests for RAG Chat & Session Persistence
# -------------------------------------------------------------

def test_chat_insufficient_evidence_when_no_documents_indexed(client):
    # Query when no matching documents exist
    res = client.post(
        "/api/chat",
        json={"message": "What is the secret launch date of Project Apollo?", "top_k": 3},
    )
    assert res.status_code == 200
    data = res.json()
    assert "insufficient evidence" in data["answer"].lower()
    assert data["citations"] == []
    assert data["role"] == "assistant"
    assert "session_id" in data


def test_chat_real_rag_lifecycle(client):
    # 1. Ingest sample document with known verifiable facts
    doc_text = (
        b"SIFT Operational Policy Manual.\n\n"
        b"Section 3: Windshield Claims Protocol.\n"
        b"Under motor vehicle coverage clause 4.2, the windshield-only deductible is fixed at AED 500. "
        b"Such claims do not reduce the policyholder's No Claims Discount (NCD).\n\n"
        b"Section 4: Rental Vehicle Provisions.\n"
        b"Courtesy replacement vehicles are provided for up to 7 calendar days."
    )
    files = {"file": ("motor_policy.txt", io.BytesIO(doc_text), "text/plain")}
    up = client.post("/api/documents/upload", files=files, data={"tag": "Legal"})
    assert up.status_code == 201
    doc_id = up.json()["id"]

    client.post(f"/api/documents/{doc_id}/chunks")
    client.post(f"/api/documents/{doc_id}/embeddings")
    client.post(f"/api/documents/{doc_id}/index")

    # 2. Chat Query
    chat_payload = {
        "message": "What is the deductible for motor vehicle windshield damage?",
        "top_k": 3,
    }
    chat_res = client.post("/api/chat", json=chat_payload)
    assert chat_res.status_code == 200
    data = chat_res.json()

    assert data["role"] == "assistant"
    assert "answer" in data
    assert "content" in data
    assert len(data["citations"]) >= 1

    # Verify structured citation provenance
    top_cit = data["citations"][0]
    assert top_cit["document_id"] == doc_id
    assert top_cit["source_filename"] == "motor_policy.txt"
    assert "windshield" in top_cit["passage"].lower()
    assert isinstance(top_cit["distance"], float)
    assert isinstance(top_cit["similarity_score"], float)

    session_id = data["session_id"]
    assert session_id.startswith("ses-")

    # 3. Conversational Session Turn 2 (with session_id)
    turn2_payload = {
        "session_id": session_id,
        "message": "How many days is a courtesy replacement vehicle provided?",
        "top_k": 3,
    }
    turn2_res = client.post("/api/chat", json=turn2_payload)
    assert turn2_res.status_code == 200
    turn2_data = turn2_res.json()
    assert turn2_data["session_id"] == session_id
    assert len(turn2_data["citations"]) >= 1

    # 4. Verify Session History via GET /api/chat/sessions/{session_id}
    history_res = client.get(f"/api/chat/sessions/{session_id}")
    assert history_res.status_code == 200
    h_data = history_res.json()
    assert h_data["session_id"] == session_id
    assert len(h_data["messages"]) == 4  # 2 user queries + 2 assistant responses
    assert h_data["messages"][0]["role"] == "user"
    assert h_data["messages"][1]["role"] == "assistant"
    assert h_data["messages"][2]["role"] == "user"
    assert h_data["messages"][3]["role"] == "assistant"

    # 5. List sessions
    list_res = client.get("/api/chat/sessions")
    assert list_res.status_code == 200
    sessions_list = list_res.json()
    assert any(s["session_id"] == session_id for s in sessions_list)

    # 6. Delete session
    del_res = client.delete(f"/api/chat/sessions/{session_id}")
    assert del_res.status_code == 204

    # 7. Deleted session now returns 404
    get_del_res = client.get(f"/api/chat/sessions/{session_id}")
    assert get_del_res.status_code == 404


def test_chat_document_id_filtering(client):
    # Upload Document X
    doc_x_bytes = b"Project X is an aerospace satellite communication link operating at 24 GHz."
    res_x = client.post("/api/documents/upload", files={"file": ("project_x.txt", io.BytesIO(doc_x_bytes), "text/plain")})
    id_x = res_x.json()["id"]
    client.post(f"/api/documents/{id_x}/chunks")
    client.post(f"/api/documents/{id_x}/embeddings")
    client.post(f"/api/documents/{id_x}/index")

    # Upload Document Y
    doc_y_bytes = b"Project Y is a deep-sea marine exploration probe operating at 100 kHz sonar."
    res_y = client.post("/api/documents/upload", files={"file": ("project_y.txt", io.BytesIO(doc_y_bytes), "text/plain")})
    id_y = res_y.json()["id"]
    client.post(f"/api/documents/{id_y}/chunks")
    client.post(f"/api/documents/{id_y}/embeddings")
    client.post(f"/api/documents/{id_y}/index")

    # Scoped Query restricted to id_x only
    chat_res = client.post(
        "/api/chat",
        json={"message": "Explain frequency and operations", "document_ids": [id_x], "top_k": 5},
    )
    assert chat_res.status_code == 200
    data = chat_res.json()
    for cit in data["citations"]:
        assert cit["document_id"] == id_x
        assert cit["document_id"] != id_y
