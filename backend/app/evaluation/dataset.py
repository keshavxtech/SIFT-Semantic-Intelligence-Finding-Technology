"""
SIFT — Reproducible RAG Evaluation Dataset & Corpus Fixtures
Contains:
1. Realistic, diverse test documents across technical, legal/policy, and security domains.
2. Curated benchmark queries across 4 categories:
   - Single-chunk answerable queries
   - Multi-chunk synthesis queries
   - Hard-negative distractor queries
   - Unsupported / out-of-domain queries
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

# ============================================================================
# 1. Corpus Document Definitions
# ============================================================================

EVAL_DOCUMENTS: List[Dict[str, str]] = [
    {
        "filename": "rag_system_spec.txt",
        "tag": "Technical",
        "description": "SIFT RAG Architecture and Vector Pipeline Specification",
        "content": (
            "SIFT SYSTEM ARCHITECTURE & RETRIEVAL SPECIFICATION\n\n"
            "Section 1.1: Document Parsing & Text Ingestion\n"
            "SIFT ingests PDF and plain text documents using PyMuPDF and UTF-8 fallback decoders. "
            "Every ingested document is fingerprinted with a SHA-256 cryptographic checksum to prevent "
            "duplicate storage and guarantee provenance integrity.\n\n"
            "Section 1.2: Structure-Aware Chunking Pipeline\n"
            "Document text is processed through sentence-boundary-preserving chunking. "
            "The default chunk size is 1000 tokens with a configurable sliding-window overlap of 150 tokens. "
            "The minimum permissible chunk size is 100 tokens to avoid semantically orphaned fragments. "
            "Each generated chunk carries deterministic 0-based sequence indices and 1-based page coordinates.\n\n"
            "Section 2.1: Dense Vector Embeddings\n"
            "SIFT uses local sentence-transformers/all-MiniLM-L6-v2 to encode text chunks into 384-dimensional dense vectors. "
            "All generated embeddings are normalized to unit L2 length to enable efficient cosine similarity comparison. "
            "Model dimension mismatches are strictly validated before indexing.\n\n"
            "Section 2.2: Persistent Vector Storage with ChromaDB\n"
            "Normalized chunk vectors and their metadata are indexed into a persistent ChromaDB collection named "
            "'sift_document_embeddings'. Cosine distance is used as the distance metric, where a distance of 0.0 "
            "represents identical vectors and distances approaching 2.0 indicate diametric opposition.\n"
        ),
    },
    {
        "filename": "motor_insurance_policy.txt",
        "tag": "Legal",
        "description": "Motor Vehicle Insurance Settlement Guidelines and Deductible Schedule",
        "content": (
            "MOTOR VEHICLE CLAIMS SETTLEMENT POLICY & DEDUCTIBLE SCHEDULE\n\n"
            "Clause 4.1: Standard Accidental Damage Deductible\n"
            "For all standard accidental collision and impact claims involving the insured vehicle, "
            "the policyholder shall bear a mandatory deductible of 1.0% of the declared vehicle value, "
            "subject to an absolute minimum excess of AED 1,000 per claim incident.\n\n"
            "Clause 4.2: Isolated Windshield Damage Deductible\n"
            "Where damage is strictly confined to the vehicle windshield with no other bodily or structural damage, "
            "a reduced fixed deductible of AED 500 applies. Furthermore, isolated windshield claims do not affect "
            "the insured's No Claims Discount (NCD), provided that no more than one windshield claim is filed per policy year. "
            "If subsequent windshield claims occur within the same policy year, standard deductible rules apply.\n\n"
            "Clause 4.3: Depreciation Schedule for Wear Components\n"
            "Replacement parts subject to operating friction, including rubber door moldings, wiper blades, and tires, "
            "are depreciated at a rate of 20% per annum from vehicle manufacture date, capped at a maximum of 60% depreciation.\n"
        ),
    },
    {
        "filename": "compliance_security_guide.txt",
        "tag": "Security",
        "description": "Enterprise Information Security and Cryptographic Storage Guide",
        "content": (
            "ENTERPRISE INFORMATION SECURITY & COMPLIANCE STANDARD\n\n"
            "Standard 101: Cryptographic Protection at Rest and in Transit\n"
            "All structured metadata, chunk embeddings, and customer payloads stored on disk must be encrypted using "
            "AES-256-GCM with hardware-backed KMS key rotation. All external network communications must mandate TLS 1.3 "
            "with forward secrecy cipher suites. TLS 1.0 and 1.1 connections are terminated at edge proxies.\n\n"
            "Standard 102: Data Retention & Audit Log Archival\n"
            "Operational audit logs, user query records, and retrieval telemetry must be retained for a mandatory duration "
            "of exactly 7 years to comply with statutory financial oversight regulations. Upon expiration of the 7-year retention period, "
            "records must undergo cryptographically verified permanent erasure.\n\n"
            "Standard 103: Incident Response and Regulatory Notification\n"
            "In the event of a confirmed personal data breach or exfiltration incident, the Data Protection Officer must notify "
            "the competent supervisory authority within a strict window of 72 hours in accordance with GDPR Article 33.\n"
        ),
    },
]

# ============================================================================
# 2. Benchmark Query Schema & Test Cases
# ============================================================================

class BenchmarkCase(BaseModel):
    id: str = Field(..., description="Unique benchmark case identifier")
    category: str = Field(..., description="Category: single_chunk, multi_chunk, hard_negative, unsupported")
    query: str = Field(..., description="Natural language user query")
    is_answerable: bool = Field(default=True, description="Whether the corpus contains sufficient evidence")
    target_documents: List[str] = Field(default_factory=list, description="Target document filenames containing evidence")
    expected_chunk_anchors: List[str] = Field(default_factory=list, description="Distinct substrings that identify relevant chunks")
    reference_facts: List[str] = Field(default_factory=list, description="Core factual entities expected in faithful answer")
    negative_distractors: List[str] = Field(default_factory=list, description="Text from similar but irrelevant passages")
    description: str = Field(default="", description="Evaluation rationale and notes")


EVAL_BENCHMARK_CASES: List[BenchmarkCase] = [
    # ------------------------------------------------------------------------
    # Category 1: Single-Chunk Answerable Queries
    # ------------------------------------------------------------------------
    BenchmarkCase(
        id="eval-q01",
        category="single_chunk",
        query="What is the deductible for isolated windshield-only damage on a motor vehicle?",
        is_answerable=True,
        target_documents=["motor_insurance_policy.txt"],
        expected_chunk_anchors=["reduced fixed deductible of AED 500", "No Claims Discount"],
        reference_facts=["AED 500", "windshield", "No Claims Discount"],
        description="Verifies retrieval of specific Clause 4.2 deductible and exemption terms.",
    ),
    BenchmarkCase(
        id="eval-q02",
        category="single_chunk",
        query="What embedding model and vector dimension does SIFT use?",
        is_answerable=True,
        target_documents=["rag_system_spec.txt"],
        expected_chunk_anchors=["all-MiniLM-L6-v2", "384-dimensional dense vectors"],
        reference_facts=["all-MiniLM-L6-v2", "384-dimensional"],
        description="Verifies retrieval of Section 2.1 technical embedding specifications.",
    ),
    BenchmarkCase(
        id="eval-q03",
        category="single_chunk",
        query="How long must enterprise audit logs be retained under compliance policy?",
        is_answerable=True,
        target_documents=["compliance_security_guide.txt"],
        expected_chunk_anchors=["mandatory duration of exactly 7 years"],
        reference_facts=["7 years", "retention"],
        description="Verifies retrieval of Standard 102 retention duration.",
    ),

    # ------------------------------------------------------------------------
    # Category 2: Multi-Chunk Synthesis Queries
    # ------------------------------------------------------------------------
    BenchmarkCase(
        id="eval-q04",
        category="multi_chunk",
        query="Explain SIFT's chunking configuration including chunk size, overlap, and minimum limits.",
        is_answerable=True,
        target_documents=["rag_system_spec.txt"],
        expected_chunk_anchors=["1000 tokens", "150 tokens", "100 tokens"],
        reference_facts=["1000 tokens", "150 tokens", "100 tokens"],
        description="Verifies retrieval of sliding-window boundaries and minimum chunk limits.",
    ),
    BenchmarkCase(
        id="eval-q05",
        category="multi_chunk",
        query="What encryption algorithms are required for data at rest and external network traffic?",
        is_answerable=True,
        target_documents=["compliance_security_guide.txt"],
        expected_chunk_anchors=["AES-256-GCM", "TLS 1.3"],
        reference_facts=["AES-256-GCM", "TLS 1.3"],
        description="Verifies multi-point extraction of cryptographic storage and transport standards.",
    ),

    # ------------------------------------------------------------------------
    # Category 3: Hard-Negative Distractor Challenge
    # ------------------------------------------------------------------------
    BenchmarkCase(
        id="eval-q06",
        category="hard_negative",
        query="What is the standard collision deductible when damage is NOT isolated to the windshield?",
        is_answerable=True,
        target_documents=["motor_insurance_policy.txt"],
        expected_chunk_anchors=["1.0% of the declared vehicle value", "minimum excess of AED 1,000"],
        reference_facts=["1.0%", "AED 1,000"],
        negative_distractors=["AED 500"],
        description="Tests whether standard collision (AED 1,000 / 1%) ranks above windshield (AED 500) for general collision.",
    ),
    BenchmarkCase(
        id="eval-q07",
        category="hard_negative",
        query="What happens to deprecated TLS 1.0 and 1.1 connections?",
        is_answerable=True,
        target_documents=["compliance_security_guide.txt"],
        expected_chunk_anchors=["TLS 1.0 and 1.1 connections are terminated at edge proxies"],
        reference_facts=["terminated", "edge proxies"],
        negative_distractors=["AES-256-GCM"],
        description="Tests precision when querying legacy TLS deprecation rather than modern TLS 1.3.",
    ),

    # ------------------------------------------------------------------------
    # Category 4: Unsupported / Out-of-Domain Queries
    # ------------------------------------------------------------------------
    BenchmarkCase(
        id="eval-q08",
        category="unsupported",
        query="What is the insurance deductible for commercial maritime yacht collisions?",
        is_answerable=False,
        target_documents=[],
        expected_chunk_anchors=[],
        reference_facts=[],
        description="Maritime yacht insurance is not covered in motor or technical policies; expects honest refusal.",
    ),
    BenchmarkCase(
        id="eval-q09",
        category="unsupported",
        query="What are the tax filing deadlines and penalty rates for corporations in Singapore?",
        is_answerable=False,
        target_documents=[],
        expected_chunk_anchors=[],
        reference_facts=[],
        description="Tax law is completely out-of-domain; tests refusal without hallucination.",
    ),
    BenchmarkCase(
        id="eval-q10",
        category="unsupported",
        query="How does quantum key distribution integrate with satellite laser links under RFC 9999?",
        is_answerable=False,
        target_documents=[],
        expected_chunk_anchors=[],
        reference_facts=[],
        description="Fictitious quantum protocol query; tests refusal and 0 citations.",
    ),
]
