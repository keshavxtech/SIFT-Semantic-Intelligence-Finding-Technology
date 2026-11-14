"""
SIFT — Phase 6 Evaluation & Quality Benchmarking Module
"""

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

__all__ = [
    "BenchmarkCase",
    "EVAL_BENCHMARK_CASES",
    "EVAL_DOCUMENTS",
    "compute_precision_at_k",
    "compute_recall_at_k",
    "compute_reciprocal_rank",
    "compute_dcg_at_k",
    "compute_idcg_at_k",
    "compute_ndcg_at_k",
    "compute_citation_correctness",
    "compute_citation_coverage",
    "compute_fact_recall",
    "evaluate_unsupported_refusal",
    "RAGEvaluationRunner",
]
