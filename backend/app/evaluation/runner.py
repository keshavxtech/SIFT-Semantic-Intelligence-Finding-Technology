"""
SIFT — RAG Evaluation Runner & Benchmark Reporter
Executes the reproducible benchmark dataset against the real SIFT retrieval
and generation pipelines, computes deterministic quality metrics, and exports
structured JSON and CSV benchmark reports.
"""

import asyncio
import csv
import io
import json
import os

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["ANONYMIZED_TELEMETRY"] = "False"
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from app.core.config import settings
from app.core.logging import logger
from app.evaluation.dataset import EVAL_BENCHMARK_CASES, EVAL_DOCUMENTS, BenchmarkCase
from app.evaluation.generation_metrics import (
    compute_citation_correctness,
    compute_citation_coverage,
    compute_fact_recall,
    evaluate_unsupported_refusal,
)
from app.evaluation.retrieval_metrics import (
    compute_ndcg_at_k,
    compute_precision_at_k,
    compute_recall_at_k,
    compute_reciprocal_rank,
)
from app.rag.generation.base import BaseLLMProvider
from app.rag.generation.mock_provider import MockLLMProvider
from app.rag.retrieval.retriever import semantic_retriever
from app.repositories.chunk_repository import chunk_repository
from app.repositories.document_repository import document_repository
from app.schemas.chat import ChatRequest
from app.services.chat_service import chat_service
from app.services.chunking_service import chunking_service
from app.services.document_service import document_service
from app.rag.embeddings.embedding_service import embedding_service
from app.services.indexing_service import indexing_service
from fastapi import UploadFile


class BenchmarkCorpusUploadFile(UploadFile):
    """
    Subclasses UploadFile to provide non-blocking in-memory read for
    evaluation corpus ingestion without AnyIO threadpool deadlocks.
    """
    def __init__(self, filename: str, content: bytes):
        super().__init__(file=io.BytesIO(content), filename=filename)
        self._raw_content = content

    async def read(self, size: int = -1) -> bytes:
        return self._raw_content


class RAGEvaluationRunner:
    """
    Orchestrates the evaluation of SIFT's retrieval and answer generation pipelines.
    Runs deterministically without external paid API dependencies.
    """

    def __init__(
        self,
        top_k: int = 5,
        dataset_version: str = "1.0.0",
        reports_dir: Optional[str] = None,
        llm_provider: Optional[BaseLLMProvider] = None,
    ):
        self.top_k = top_k
        self.dataset_version = dataset_version
        self.reports_dir = Path(reports_dir or os.path.join(settings.storage_path.parent, "reports"))
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.llm_provider = llm_provider or MockLLMProvider()

    async def ensure_evaluation_corpus(self) -> Dict[str, str]:
        """
        Ensures all documents in the evaluation dataset are ingested, chunked,
        embedded, and indexed into the active ChromaDB collection.
        Returns a mapping of filename -> document_id.
        """
        filename_to_id: Dict[str, str] = {}
        all_docs = document_repository.list_all()

        for doc_def in EVAL_DOCUMENTS:
            fname = doc_def["filename"]
            existing = next((d for d in all_docs if d.original_filename == fname), None)

            if existing:
                doc_id = existing.id
                status = await indexing_service.get_index_status(doc_id)
                if not status.is_indexed:
                    logger.info(f"Indexing existing unindexed eval document: {fname} ({doc_id})")
                    chunking_service.chunk_document(doc_id)
                    await embedding_service.generate_document_embeddings(doc_id)
                    await indexing_service.index_document(doc_id)
                filename_to_id[fname] = doc_id
            else:
                logger.info(f"Ingesting evaluation corpus document: {fname}")
                content_bytes = doc_def["content"].encode("utf-8")
                upload_file = BenchmarkCorpusUploadFile(
                    filename=fname,
                    content=content_bytes,
                )
                doc = await document_service.upload_document(
                    file=upload_file,
                    tag=doc_def.get("tag", "Technical"),
                    description=doc_def.get("description", "Evaluation Benchmark Corpus"),
                )
                doc_id = doc.id
                chunking_service.chunk_document(doc_id)
                await embedding_service.generate_document_embeddings(doc_id)
                await indexing_service.index_document(doc_id)
                filename_to_id[fname] = doc_id

        return filename_to_id

    def resolve_expected_chunks(
        self,
        case: BenchmarkCase,
        filename_to_id: Dict[str, str],
    ) -> Set[str]:
        """
        Deterministically resolves expected relevant chunk IDs by matching expected
        textual anchors against the actual persisted chunks of target documents.
        """
        expected_ids: Set[str] = set()
        if not case.is_answerable or not case.target_documents:
            return expected_ids

        for target_fname in case.target_documents:
            doc_id = filename_to_id.get(target_fname)
            if not doc_id:
                continue

            chunks = chunk_repository.get_chunks_by_document(doc_id)
            for chunk in chunks:
                chunk_text = chunk.content.lower()
                for anchor in case.expected_chunk_anchors:
                    if anchor.lower() in chunk_text:
                        expected_ids.add(chunk.chunk_id)
                        break

        return expected_ids

    async def run_evaluation(
        self,
        cases: Optional[List[BenchmarkCase]] = None,
    ) -> Dict[str, Any]:
        """
        Executes the benchmark evaluation over all query cases.
        """
        cases_to_run = cases or EVAL_BENCHMARK_CASES
        filename_to_id = await self.ensure_evaluation_corpus()

        # Temporarily use deterministic evaluation LLM provider
        original_provider = chat_service.provider
        chat_service.provider = self.llm_provider

        case_results: List[Dict[str, Any]] = []
        start_eval_time = time.perf_counter()

        try:
            for case in cases_to_run:
                logger.info(f"Evaluating case {case.id} [{case.category}]: {case.query[:50]}...")
                expected_chunk_ids = self.resolve_expected_chunks(case, filename_to_id)

                # 1. Retrieval Phase
                retrieval_start = time.perf_counter()
                doc_ids = (
                    [filename_to_id[f] for f in case.target_documents if f in filename_to_id]
                    if case.is_answerable
                    else ["doc_out_of_scope"]
                )
                raw_passages = semantic_retriever.retrieve(
                    query=case.query,
                    document_ids=doc_ids,
                    top_k=self.top_k,
                    mode="dense",
                )
                retrieval_latency_ms = (time.perf_counter() - retrieval_start) * 1000
                retrieved_ids = [p["chunk_id"] for p in raw_passages]

                # Compute Retrieval Metrics
                prec = compute_precision_at_k(retrieved_ids, expected_chunk_ids, k=self.top_k)
                rec = compute_recall_at_k(retrieved_ids, expected_chunk_ids, k=self.top_k)
                rr = compute_reciprocal_rank(retrieved_ids, expected_chunk_ids, k=self.top_k)
                ndcg = compute_ndcg_at_k(retrieved_ids, expected_chunk_ids, k=self.top_k)

                # 2. Generation & Citation Phase
                gen_start = time.perf_counter()
                chat_req = ChatRequest(
                    message=case.query,
                    document_ids=doc_ids,
                    top_k=self.top_k,
                    retrieval_mode="dense",
                )
                chat_res = await chat_service.generate_chat_response(chat_req)
                generation_latency_ms = (time.perf_counter() - gen_start) * 1000

                cited_ids = [c.chunk_id for c in chat_res.citations]
                answer = chat_res.answer

                # Compute Generation & Citation Metrics
                cit_correctness = compute_citation_correctness(
                    cited_ids, retrieved_ids, is_answerable=case.is_answerable
                )
                cit_coverage = compute_citation_coverage(
                    cited_ids, retrieved_ids, expected_chunk_ids, is_answerable=case.is_answerable
                )
                fact_recall = compute_fact_recall(answer, case.reference_facts)
                refusal_acc = evaluate_unsupported_refusal(
                    answer, cited_ids, is_answerable=case.is_answerable
                )

                total_latency_ms = retrieval_latency_ms + generation_latency_ms

                # Clean up the session generated for this eval query
                try:
                    chat_service.delete_session(chat_res.session_id)
                except Exception:
                    pass

                case_results.append({
                    "case_id": case.id,
                    "category": case.category,
                    "query": case.query,
                    "is_answerable": case.is_answerable,
                    "expected_chunks_count": len(expected_chunk_ids),
                    "retrieved_chunks_count": len(retrieved_ids),
                    "cited_chunks_count": len(cited_ids),
                    "precision_at_k": round(prec, 4),
                    "recall_at_k": round(rec, 4),
                    "reciprocal_rank": round(rr, 4),
                    "ndcg_at_k": round(ndcg, 4),
                    "citation_correctness": round(cit_correctness, 4),
                    "citation_coverage": round(cit_coverage, 4),
                    "fact_recall": round(fact_recall, 4),
                    "unsupported_refusal_accuracy": round(refusal_acc, 4),
                    "retrieval_latency_ms": round(retrieval_latency_ms, 2),
                    "generation_latency_ms": round(generation_latency_ms, 2),
                    "total_latency_ms": round(total_latency_ms, 2),
                    "answer_preview": answer[:140] + "..." if len(answer) > 140 else answer,
                })

        finally:
            chat_service.provider = original_provider

        total_eval_duration_s = time.perf_counter() - start_eval_time

        # 3. Compute Aggregate Benchmark Summary
        num_cases = len(case_results)
        answerable_cases = [r for r in case_results if r["is_answerable"]]
        unsupported_cases = [r for r in case_results if not r["is_answerable"]]

        mean_precision = sum(r["precision_at_k"] for r in answerable_cases) / len(answerable_cases) if answerable_cases else 1.0
        mean_recall = sum(r["recall_at_k"] for r in answerable_cases) / len(answerable_cases) if answerable_cases else 1.0
        mrr = sum(r["reciprocal_rank"] for r in answerable_cases) / len(answerable_cases) if answerable_cases else 1.0
        mean_ndcg = sum(r["ndcg_at_k"] for r in answerable_cases) / len(answerable_cases) if answerable_cases else 1.0

        mean_cit_correctness = sum(r["citation_correctness"] for r in case_results) / num_cases if num_cases else 1.0
        mean_cit_coverage = sum(r["citation_coverage"] for r in answerable_cases) / len(answerable_cases) if answerable_cases else 1.0
        mean_fact_recall = sum(r["fact_recall"] for r in answerable_cases) / len(answerable_cases) if answerable_cases else 1.0
        refusal_rate = sum(r["unsupported_refusal_accuracy"] for r in unsupported_cases) / len(unsupported_cases) if unsupported_cases else 1.0

        mean_latency = sum(r["total_latency_ms"] for r in case_results) / num_cases if num_cases else 0.0

        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")

        report_summary: Dict[str, Any] = {
            "evaluation_metadata": {
                "dataset_version": self.dataset_version,
                "timestamp": datetime.now().isoformat(),
                "top_k": self.top_k,
                "llm_provider": self.llm_provider.provider_name,
                "llm_model": self.llm_provider.model_name,
                "embedding_model": settings.DEFAULT_EMBEDDING_MODEL,
                "vector_dimension": 384,
                "distance_metric": "cosine",
                "total_cases": num_cases,
                "answerable_cases": len(answerable_cases),
                "unsupported_cases": len(unsupported_cases),
                "total_eval_duration_seconds": round(total_eval_duration_s, 2),
            },
            "aggregate_metrics": {
                "mean_precision_at_k": round(mean_precision, 4),
                "mean_recall_at_k": round(mean_recall, 4),
                "mean_reciprocal_rank": round(mrr, 4),
                "mean_ndcg_at_k": round(mean_ndcg, 4),
                "mean_citation_correctness": round(mean_cit_correctness, 4),
                "mean_citation_coverage": round(mean_cit_coverage, 4),
                "mean_fact_recall": round(mean_fact_recall, 4),
                "unsupported_refusal_accuracy": round(refusal_rate, 4),
                "average_latency_ms": round(mean_latency, 2),
            },
            "case_results": case_results,
        }

        # 4. Export Structured Reports
        json_path = self.reports_dir / f"rag_eval_report_{timestamp_str}.json"
        csv_path = self.reports_dir / f"rag_eval_summary_{timestamp_str}.csv"

        with open(json_path, "w", encoding="utf-8") as jf:
            json.dump(report_summary, jf, indent=2)

        with open(csv_path, "w", newline="", encoding="utf-8") as cf:
            writer = csv.DictWriter(
                cf,
                fieldnames=[
                    "case_id",
                    "category",
                    "query",
                    "is_answerable",
                    "precision_at_k",
                    "recall_at_k",
                    "reciprocal_rank",
                    "ndcg_at_k",
                    "citation_correctness",
                    "citation_coverage",
                    "fact_recall",
                    "unsupported_refusal_accuracy",
                    "retrieval_latency_ms",
                    "generation_latency_ms",
                    "total_latency_ms",
                ],
            )
            writer.writeheader()
            for r in case_results:
                row = {k: v for k, v in r.items() if k != "answer_preview" and k != "expected_chunks_count" and k != "retrieved_chunks_count" and k != "cited_chunks_count"}
                writer.writerow(row)

        report_summary["exported_files"] = {
            "json_report": str(json_path.resolve()),
            "csv_summary": str(csv_path.resolve()),
        }

        logger.info(f"Evaluation report generated successfully: {json_path}")
        return report_summary


# CLI Execution Support
if __name__ == "__main__":
    runner = RAGEvaluationRunner()
    result = asyncio.run(runner.run_evaluation())
    print("\n" + "=" * 65)
    print("SIFT RAG QUALITY BENCHMARK REPORT")
    print("=" * 65)
    print(f"Dataset Version: {result['evaluation_metadata']['dataset_version']}")
    print(f"Total Test Cases: {result['evaluation_metadata']['total_cases']}")
    print(f"Mean Precision@{result['evaluation_metadata']['top_k']}: {result['aggregate_metrics']['mean_precision_at_k'] * 100:.2f}%")
    print(f"Mean Recall@{result['evaluation_metadata']['top_k']}:    {result['aggregate_metrics']['mean_recall_at_k'] * 100:.2f}%")
    print(f"Mean Reciprocal Rank (MRR):  {result['aggregate_metrics']['mean_reciprocal_rank']:.4f}")
    print(f"Mean nDCG@{result['evaluation_metadata']['top_k']}:        {result['aggregate_metrics']['mean_ndcg_at_k']:.4f}")
    print(f"Citation Correctness:        {result['aggregate_metrics']['mean_citation_correctness'] * 100:.2f}%")
    print(f"Citation Coverage:           {result['aggregate_metrics']['mean_citation_coverage'] * 100:.2f}%")
    print(f"Fact Recall:                 {result['aggregate_metrics']['mean_fact_recall'] * 100:.2f}%")
    print(f"Unsupported Refusal Acc:     {result['aggregate_metrics']['unsupported_refusal_accuracy'] * 100:.2f}%")
    print(f"Average Total Latency:       {result['aggregate_metrics']['average_latency_ms']:.2f} ms")
    print("-" * 65)
    print(f"JSON Report: {result['exported_files']['json_report']}")
    print(f"CSV Summary: {result['exported_files']['csv_summary']}")
    print("=" * 65 + "\n")
