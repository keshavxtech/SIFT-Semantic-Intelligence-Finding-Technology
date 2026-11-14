"""
SIFT — Answer and Citation Quality Evaluation Metrics
Provides deterministic evaluations for:
- Citation Correctness (citations strictly present in retrieved context)
- Citation Coverage (retrieved evidence chunks cited in answer)
- Answer Fact Recall & Faithfulness
- Unsupported-Question Refusal Accuracy
"""

from typing import Collection, List, Sequence, Set


def compute_citation_correctness(
    cited_chunk_ids: Sequence[str],
    retrieved_chunk_ids: Collection[str],
    is_answerable: bool = True,
) -> float:
    """
    Evaluates whether every cited chunk was genuinely present in the retrieved context.

    Citation Correctness = |Cited ∩ Retrieved| / |Cited|

    Edge cases:
    - If cited_chunk_ids is empty:
      - returns 1.0 if query was unsupported (no citations expected)
      - returns 0.0 if query was answerable (expected citations)
    """
    if not cited_chunk_ids:
        return 1.0 if not is_answerable else 0.0

    retrieved_set: Set[str] = set(retrieved_chunk_ids)
    valid_citations = sum(1 for cid in cited_chunk_ids if cid in retrieved_set)
    return valid_citations / float(len(cited_chunk_ids))


def compute_citation_coverage(
    cited_chunk_ids: Sequence[str],
    retrieved_chunk_ids: Sequence[str],
    expected_relevant_ids: Collection[str],
    is_answerable: bool = True,
) -> float:
    """
    Evaluates whether retrieved chunks containing ground-truth evidence were actually cited.

    Citation Coverage = |Cited ∩ (Retrieved ∩ Relevant)| / |Retrieved ∩ Relevant|

    Edge cases:
    - If |Retrieved ∩ Relevant| == 0:
      - returns 1.0 if query was unsupported
      - returns 0.0 if query was answerable but no relevant chunk was retrieved
    """
    if not is_answerable:
        return 1.0 if not cited_chunk_ids else 0.0

    retrieved_set = set(retrieved_chunk_ids)
    relevant_set = set(expected_relevant_ids)
    retrieved_relevant = retrieved_set.intersection(relevant_set)

    if not retrieved_relevant:
        return 0.0

    cited_set = set(cited_chunk_ids)
    covered = sum(1 for cid in retrieved_relevant if cid in cited_set)
    return covered / float(len(retrieved_relevant))


def compute_fact_recall(
    answer: str,
    expected_facts: Sequence[str],
) -> float:
    """
    Computes factual containment: the fraction of key expected factual assertions
    that are present (case-insensitively) in the generated answer.

    Edge cases:
    - If expected_facts is empty: returns 1.0
    - If answer is empty: returns 0.0
    """
    if not expected_facts:
        return 1.0

    if not answer or not answer.strip():
        return 0.0

    normalized_answer = answer.lower()
    matched = 0
    for fact in expected_facts:
        if fact.lower() in normalized_answer:
            matched += 1

    return matched / float(len(expected_facts))


def evaluate_unsupported_refusal(
    answer: str,
    cited_chunk_ids: Sequence[str],
    is_answerable: bool,
) -> float:
    """
    Evaluates whether an unsupported question was handled truthfully:
    - Answer must clearly state insufficient evidence.
    - Zero hallucinated citations should be produced.

    Returns:
    - 1.0 if correctly refused or if question was actually answerable
    - 0.0 if an unsupported question produced an ungrounded answer or citations
    """
    if is_answerable:
        return 1.0

    refusal_cues = [
        "insufficient evidence",
        "not mentioned",
        "not provided",
        "does not contain",
        "cannot be answered",
        "no information",
        "unable to determine",
    ]

    answer_lower = (answer or "").lower()
    has_refusal_phrase = any(cue in answer_lower for cue in refusal_cues)
    has_zero_citations = len(cited_chunk_ids) == 0

    return 1.0 if (has_refusal_phrase and has_zero_citations) else 0.0
