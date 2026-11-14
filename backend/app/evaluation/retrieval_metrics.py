"""
SIFT — Deterministic Retrieval Evaluation Metrics
Provides mathematical implementations of:
- Precision@K
- Recall@K
- Reciprocal Rank (RR) / Mean Reciprocal Rank (MRR)
- Normalized Discounted Cumulative Gain (nDCG@K)
"""

import math
from typing import Collection, List, Sequence, Set, Union


def compute_precision_at_k(
    retrieved_ids: Sequence[str],
    relevant_ids: Collection[str],
    k: int,
) -> float:
    """
    Computes Precision@K: the fraction of retrieved chunks in the top-K that are relevant.

    Precision@K = |Retrieved[:K] ∩ Relevant| / K

    Edge cases:
    - If k <= 0: returns 0.0
    - If retrieved_ids is empty: returns 0.0
    - If relevant_ids is empty:
      - returns 1.0 if retrieved_ids is also empty
      - returns 0.0 if chunks were retrieved when none were relevant
    """
    if k <= 0:
        return 0.0

    rel_set: Set[str] = set(relevant_ids)
    if not rel_set:
        return 1.0 if not retrieved_ids else 0.0

    top_k = retrieved_ids[:k]
    if not top_k:
        return 0.0

    relevant_retrieved = sum(1 for cid in top_k if cid in rel_set)
    return relevant_retrieved / float(k)


def compute_recall_at_k(
    retrieved_ids: Sequence[str],
    relevant_ids: Collection[str],
    k: int,
) -> float:
    """
    Computes Recall@K: the fraction of expected relevant chunks successfully retrieved in the top-K.

    Recall@K = |Retrieved[:K] ∩ Relevant| / |Relevant|

    Edge cases:
    - If relevant_ids is empty: returns 1.0 (all 0 expected items recovered)
    - If k <= 0 or retrieved_ids is empty: returns 0.0
    """
    rel_set: Set[str] = set(relevant_ids)
    if not rel_set:
        return 1.0

    if k <= 0 or not retrieved_ids:
        return 0.0

    top_k = retrieved_ids[:k]
    relevant_retrieved = sum(1 for cid in top_k if cid in rel_set)
    return relevant_retrieved / float(len(rel_set))


def compute_reciprocal_rank(
    retrieved_ids: Sequence[str],
    relevant_ids: Collection[str],
    k: int = 10,
) -> float:
    """
    Computes Reciprocal Rank (RR): reciprocal of the 1-based rank of the first relevant item.

    RR = 1 / rank if first relevant item is found at rank <= k, else 0.0
    """
    if k <= 0 or not retrieved_ids:
        return 0.0

    rel_set: Set[str] = set(relevant_ids)
    if not rel_set:
        return 0.0

    top_k = retrieved_ids[:k]
    for idx, cid in enumerate(top_k, start=1):
        if cid in rel_set:
            return 1.0 / float(idx)

    return 0.0


def compute_dcg_at_k(
    retrieved_ids: Sequence[str],
    relevant_ids: Collection[str],
    k: int,
) -> float:
    """
    Computes Discounted Cumulative Gain (DCG@K) using standard logarithmic discounting:
    DCG@K = sum_{i=1..min(k, len(retrieved))} (rel_i / log2(i + 1))
    """
    if k <= 0 or not retrieved_ids:
        return 0.0

    rel_set: Set[str] = set(relevant_ids)
    if not rel_set:
        return 0.0

    dcg = 0.0
    top_k = retrieved_ids[:k]
    for idx, cid in enumerate(top_k, start=1):
        if cid in rel_set:
            dcg += 1.0 / math.log2(idx + 1)

    return dcg


def compute_idcg_at_k(
    relevant_count: int,
    k: int,
) -> float:
    """
    Computes Ideal DCG (IDCG@K) for binary relevance:
    IDCG@K = sum_{i=1..min(k, relevant_count)} (1 / log2(i + 1))
    """
    if k <= 0 or relevant_count <= 0:
        return 0.0

    ideal_hits = min(k, relevant_count)
    return sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))


def compute_ndcg_at_k(
    retrieved_ids: Sequence[str],
    relevant_ids: Collection[str],
    k: int,
) -> float:
    """
    Computes Normalized Discounted Cumulative Gain (nDCG@K):
    nDCG@K = DCG@K / IDCG@K

    Edge cases:
    - If IDCG@K == 0:
      - returns 1.0 if DCG@K is also 0 (e.g. no relevant items existed)
      - returns 0.0 otherwise
    """
    if k <= 0:
        return 0.0

    rel_set: Set[str] = set(relevant_ids)
    idcg = compute_idcg_at_k(len(rel_set), k)
    dcg = compute_dcg_at_k(retrieved_ids, rel_set, k)

    if idcg == 0.0:
        return 1.0 if not retrieved_ids else 0.0

    return dcg / idcg
