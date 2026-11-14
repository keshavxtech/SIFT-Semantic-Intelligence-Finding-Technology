"""
Tests for SIFT Phase 6: RAG Evaluation & Quality Benchmarking
Validates:
- Precision@K and Recall@K mathematical calculations
- Reciprocal Rank (RR) and nDCG@K calculations
- Edge case handling (empty retrieved, empty relevant, k <= 0)
- Citation correctness and coverage calculations
- Fact recall calculation
- Unsupported-question refusal evaluation
- Evaluation runner execution and report file exports
- Metric reproducibility across multiple runs
"""

import json
import math
import os
import pytest
from app.evaluation.dataset import (
    BenchmarkCase,
    EVAL_BENCHMARK_CASES,
    EVAL_DOCUMENTS,
)
from app.evaluation.generation_metrics import (
    compute_citation_correctness,
    compute_citation_coverage,
    compute_fact_recall,
    evaluate_unsupported_refusal,
)
from app.evaluation.retrieval_metrics import (
    compute_dcg_at_k,
    compute_idcg_at_k,
    compute_ndcg_at_k,
    compute_precision_at_k,
    compute_recall_at_k,
    compute_reciprocal_rank,
)
from app.evaluation.runner import RAGEvaluationRunner
from app.rag.generation.mock_provider import MockLLMProvider


# ============================================================================
# 1. Retrieval Metrics Unit Tests
# ============================================================================

def test_precision_at_k_standard():
    retrieved = ["chk_1", "chk_2", "chk_3", "chk_4", "chk_5"]
    relevant = {"chk_1", "chk_3"}
    # Top-5 has 2 relevant -> 2/5 = 0.4
    assert compute_precision_at_k(retrieved, relevant, k=5) == 0.4
    # Top-2 has 1 relevant ("chk_1") -> 1/2 = 0.5
    assert compute_precision_at_k(retrieved, relevant, k=2) == 0.5
    # Top-1 has 1 relevant -> 1/1 = 1.0
    assert compute_precision_at_k(retrieved, relevant, k=1) == 1.0


def test_precision_at_k_edge_cases():
    # k <= 0
    assert compute_precision_at_k(["chk_1"], {"chk_1"}, k=0) == 0.0
    assert compute_precision_at_k(["chk_1"], {"chk_1"}, k=-2) == 0.0

    # Empty retrieved
    assert compute_precision_at_k([], {"chk_1"}, k=5) == 0.0

    # Empty relevant (unsupported question)
    assert compute_precision_at_k([], set(), k=5) == 1.0
    assert compute_precision_at_k(["chk_1"], set(), k=5) == 0.0


def test_recall_at_k_standard():
    retrieved = ["chk_1", "chk_2", "chk_3", "chk_4", "chk_5"]
    relevant = {"chk_1", "chk_3", "chk_9"}  # 3 relevant chunks
    # Top-3 has 2 of 3 relevant -> 2/3 ≈ 0.6667
    assert pytest.approx(compute_recall_at_k(retrieved, relevant, k=3), rel=1e-3) == 2.0 / 3.0
    # Top-1 has 1 of 3 relevant -> 1/3 ≈ 0.3333
    assert pytest.approx(compute_recall_at_k(retrieved, relevant, k=1), rel=1e-3) == 1.0 / 3.0
    # If all relevant retrieved in top-k
    relevant_sub = {"chk_1", "chk_2"}
    assert compute_recall_at_k(retrieved, relevant_sub, k=2) == 1.0


def test_recall_at_k_edge_cases():
    # Empty relevant
    assert compute_recall_at_k(["chk_1"], set(), k=5) == 1.0
    # Empty retrieved with non-empty relevant
    assert compute_recall_at_k([], {"chk_1"}, k=5) == 0.0
    # k <= 0
    assert compute_recall_at_k(["chk_1"], {"chk_1"}, k=0) == 0.0


def test_reciprocal_rank():
    # Relevant item at rank 1 -> RR = 1.0
    assert compute_reciprocal_rank(["chk_1", "chk_2"], {"chk_1"}) == 1.0
    # Relevant item at rank 2 -> RR = 1/2 = 0.5
    assert compute_reciprocal_rank(["chk_2", "chk_1"], {"chk_1"}) == 0.5
    # Relevant item at rank 4 -> RR = 1/4 = 0.25
    assert compute_reciprocal_rank(["chk_a", "chk_b", "chk_c", "chk_d"], {"chk_d"}) == 0.25
    # Relevant item not found in top-k
    assert compute_reciprocal_rank(["chk_a", "chk_b"], {"chk_z"}, k=2) == 0.0
    # Empty retrieved or empty relevant
    assert compute_reciprocal_rank([], {"chk_1"}) == 0.0
    assert compute_reciprocal_rank(["chk_1"], set()) == 0.0


def test_ndcg_at_k_perfect_ranking():
    # Relevant items ranked first: rank 1 and 2
    retrieved = ["chk_1", "chk_2", "chk_x", "chk_y"]
    relevant = {"chk_1", "chk_2"}
    # Perfect ranking should have nDCG = 1.0
    assert pytest.approx(compute_ndcg_at_k(retrieved, relevant, k=4), rel=1e-4) == 1.0


def test_ndcg_at_k_imperfect_ranking():
    # Relevant item at rank 2 instead of rank 1
    retrieved = ["chk_irrelevant", "chk_relevant"]
    relevant = {"chk_relevant"}
    # DCG = 1 / log2(3) ≈ 0.6309
    # IDCG = 1 / log2(2) = 1.0
    # nDCG = 0.6309 / 1.0 ≈ 0.6309
    expected_ndcg = (1.0 / math.log2(3)) / (1.0 / math.log2(2))
    assert pytest.approx(compute_ndcg_at_k(retrieved, relevant, k=2), rel=1e-4) == expected_ndcg


def test_ndcg_at_k_edge_cases():
    # Zero relevant items
    assert compute_ndcg_at_k([], set(), k=5) == 1.0
    assert compute_ndcg_at_k(["chk_1"], set(), k=5) == 0.0
    # Empty retrieved
    assert compute_ndcg_at_k([], {"chk_1"}, k=5) == 0.0


# ============================================================================
# 2. Generation & Citation Metrics Unit Tests
# ============================================================================

def test_citation_correctness():
    retrieved = ["chk_1", "chk_2", "chk_3"]
    # All cited are retrieved -> 1.0
    assert compute_citation_correctness(["chk_1", "chk_2"], retrieved) == 1.0
    # One cited is not retrieved -> 1/2 = 0.5
    assert compute_citation_correctness(["chk_1", "chk_hallucinated"], retrieved) == 0.5
    # No citations produced for answerable query -> 0.0
    assert compute_citation_correctness([], retrieved, is_answerable=True) == 0.0
    # No citations produced for unsupported query -> 1.0
    assert compute_citation_correctness([], retrieved, is_answerable=False) == 1.0


def test_citation_coverage():
    retrieved = ["chk_1", "chk_2", "chk_3"]
    expected_relevant = {"chk_1", "chk_2"}
    # Both retrieved relevant chunks cited -> 2/2 = 1.0
    assert compute_citation_coverage(["chk_1", "chk_2"], retrieved, expected_relevant) == 1.0
    # Only 1 of 2 retrieved relevant cited -> 1/2 = 0.5
    assert compute_citation_coverage(["chk_1"], retrieved, expected_relevant) == 0.5
    # Unsupported query with no citations -> 1.0
    assert compute_citation_coverage([], retrieved, expected_relevant, is_answerable=False) == 1.0


def test_fact_recall():
    answer = "The standard deductible is AED 1,000, while the windshield excess is AED 500."
    facts = ["AED 1,000", "AED 500", "windshield"]
    assert compute_fact_recall(answer, facts) == 1.0

    facts_partial = ["AED 1,000", "AED 500", "unmentioned_entity"]
    assert pytest.approx(compute_fact_recall(answer, facts_partial), rel=1e-3) == 2.0 / 3.0

    assert compute_fact_recall("", facts) == 0.0
    assert compute_fact_recall("some text", []) == 1.0


def test_unsupported_refusal_evaluation():
    # Correct refusal: mentions insufficient evidence and produced 0 citations
    honest_answer = "Based on the provided documents, there is insufficient evidence to answer your question."
    assert evaluate_unsupported_refusal(honest_answer, cited_chunk_ids=[], is_answerable=False) == 1.0

    # Failed refusal: answer fabricated claims without evidence
    fabricated_answer = "The maritime insurance deductible is 50,000 USD."
    assert evaluate_unsupported_refusal(fabricated_answer, cited_chunk_ids=[], is_answerable=False) == 0.0

    # Failed refusal: produced citations for an unsupported question
    assert evaluate_unsupported_refusal(honest_answer, cited_chunk_ids=["chk_1"], is_answerable=False) == 0.0

    # Answerable question always passes refusal check
    assert evaluate_unsupported_refusal("Any answer", cited_chunk_ids=["chk_1"], is_answerable=True) == 1.0


# ============================================================================
# 3. Dataset Integrity & Determinism Tests
# ============================================================================

def test_benchmark_dataset_structure():
    assert len(EVAL_DOCUMENTS) >= 3, "Expected at least 3 evaluation corpus documents"
    assert len(EVAL_BENCHMARK_CASES) >= 8, "Expected at least 8 benchmark query cases"

    categories = {case.category for case in EVAL_BENCHMARK_CASES}
    assert "single_chunk" in categories
    assert "multi_chunk" in categories
    assert "hard_negative" in categories
    assert "unsupported" in categories

    for case in EVAL_BENCHMARK_CASES:
        assert case.id and case.query
        if case.is_answerable:
            assert len(case.target_documents) > 0
            assert len(case.expected_chunk_anchors) > 0


def test_deterministic_metrics_reproducibility():
    retrieved = ["chk_a", "chk_b", "chk_c"]
    relevant = {"chk_a", "chk_c"}

    res1 = (
        compute_precision_at_k(retrieved, relevant, k=3),
        compute_recall_at_k(retrieved, relevant, k=3),
        compute_reciprocal_rank(retrieved, relevant, k=3),
        compute_ndcg_at_k(retrieved, relevant, k=3),
    )
    res2 = (
        compute_precision_at_k(retrieved, relevant, k=3),
        compute_recall_at_k(retrieved, relevant, k=3),
        compute_reciprocal_rank(retrieved, relevant, k=3),
        compute_ndcg_at_k(retrieved, relevant, k=3),
    )
    assert res1 == res2, "Retrieval metrics calculation must be 100% deterministic"


# ============================================================================
# 4. End-to-End Evaluation Runner Integration Test
# ============================================================================

def test_evaluation_runner_execution_and_reports(tmp_path, monkeypatch):
    # Run evaluation runner with a subset of cases in a temporary reports directory
    test_cases = [
        EVAL_BENCHMARK_CASES[0],  # Answerable single chunk (windshield AED 500)
        EVAL_BENCHMARK_CASES[-1], # Unsupported query (quantum key distribution)
    ]

    runner = RAGEvaluationRunner(
        top_k=3,
        reports_dir=str(tmp_path),
        llm_provider=MockLLMProvider(),
    )

    async def mock_ensure_corpus():
        return {
            "rag_system_spec.txt": "doc-01",
            "motor_insurance_policy.txt": "doc-02",
            "compliance_security_guide.txt": "doc-03",
        }

    from app.rag.retrieval.retriever import semantic_retriever
    from app.services.chat_service import chat_service

    monkeypatch.setattr(runner, "ensure_evaluation_corpus", mock_ensure_corpus)
    monkeypatch.setattr(
        semantic_retriever,
        "retrieve",
        lambda *args, **kwargs: [
            {"chunk_id": "chk-01", "similarity_score": 0.95, "content": "Windshield deductible is AED 500"}
        ] if kwargs.get("document_ids") != ["doc_out_of_scope"] else []
    )
    monkeypatch.setattr(
        runner,
        "resolve_expected_chunks",
        lambda case, f_map: {"chk-01"} if case.is_answerable else set()
    )

    from app.schemas.chat import ChatResponse, CitationSchema
    async def mock_generate_chat_response(*args, **kwargs):
        req = args[0] if (len(args) == 1 and not hasattr(args[0], "generate_chat_response")) else (args[1] if len(args) > 1 else kwargs.get("request"))
        if req and req.document_ids == ["doc_out_of_scope"]:
            return ChatResponse(
                id="msg-eval-01",
                session_id="ses-eval-01",
                role="assistant",
                content="Based on the provided documents, there is insufficient evidence to answer your question.",
                answer="Based on the provided documents, there is insufficient evidence to answer your question.",
                timestamp="2026-11-13T20:24:00Z",
                citations=[],
                model="mock-rag-llama-3",
                provider="mock",
            )
        return ChatResponse(
            id="msg-eval-02",
            session_id="ses-eval-02",
            role="assistant",
            content="Based on the provided documentation, windshield deductible is fixed at AED 500 [1].",
            answer="Based on the provided documentation, windshield deductible is fixed at AED 500 [1].",
            timestamp="2026-11-13T20:24:00Z",
            citations=[
                CitationSchema(
                    id="cit-01",
                    document_id="doc-02",
                    chunk_id="chk-01",
                    chunk_index=0,
                    source_filename="motor_insurance_policy.txt",
                    page_start=1,
                    page_end=1,
                    passage="windshield deductible is fixed at AED 500",
                    distance=0.05,
                    similarity_score=0.95,
                    docId="doc-02",
                    docName="motor_insurance_policy.txt",
                    pageNumber=1,
                    chunkId="chk-01",
                    similarityScore=0.95,
                    sectionTitle="Section 3",
                )
            ],
            model="mock-rag-llama-3",
            provider="mock",
        )

    monkeypatch.setattr(chat_service, "generate_chat_response", mock_generate_chat_response)
    monkeypatch.setattr(chat_service, "delete_session", lambda s_id: None)

    import asyncio
    report = asyncio.run(runner.run_evaluation(cases=test_cases))

    # Verify report structure
    assert "evaluation_metadata" in report
    assert "aggregate_metrics" in report
    assert "case_results" in report
    assert "exported_files" in report

    assert len(report["case_results"]) == 2
    assert report["evaluation_metadata"]["total_cases"] == 2
    assert report["evaluation_metadata"]["answerable_cases"] == 1
    assert report["evaluation_metadata"]["unsupported_cases"] == 1

    # Verify file exports exist on disk
    json_path = report["exported_files"]["json_report"]
    csv_path = report["exported_files"]["csv_summary"]

    assert os.path.exists(json_path), f"JSON report file was not created: {json_path}"
    assert os.path.exists(csv_path), f"CSV summary file was not created: {csv_path}"

    # Verify exported JSON is valid and matches
    with open(json_path, "r", encoding="utf-8") as f:
        loaded_json = json.load(f)
    assert loaded_json["evaluation_metadata"]["dataset_version"] == "1.0.0"
    assert len(loaded_json["case_results"]) == 2
