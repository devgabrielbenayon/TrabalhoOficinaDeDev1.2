"""Offline ranking metrics with binary relevance (a card is relevant iff it is in the deck's test set).

``recs`` is always a ranked list of item indices (best first); ``relevant`` a set of item indices.
"""

from __future__ import annotations

from typing import Collection, Sequence

import numpy as np


def precision_at_n(recs: Sequence[int], relevant: Collection[int], n: int) -> float:
    """Hits in the top ``n`` divided by ``n``."""
    rel = set(relevant)
    return sum(i in rel for i in recs[:n]) / n


def recall_at_n(recs: Sequence[int], relevant: Collection[int], n: int) -> float:
    """Hits in the top ``n`` divided by ``|relevant|`` (0 if there is nothing relevant)."""
    rel = set(relevant)
    return sum(i in rel for i in recs[:n]) / len(rel) if rel else 0.0


def dcg_at_n(recs: Sequence[int], relevant: Collection[int], n: int) -> float:
    """Sum of ``1 / log2(rank + 1)`` over the hits, rank starting at 1."""
    rel = set(relevant)
    return float(sum(1.0 / np.log2(rank + 1) for rank, i in enumerate(recs[:n], 1) if i in rel))


def ndcg_at_n(recs: Sequence[int], relevant: Collection[int], n: int) -> float:
    """DCG normalized by the ideal DCG, which has ``min(|relevant|, n)`` hits at the top."""
    n_ideal = min(len(set(relevant)), n)
    idcg = float(sum(1.0 / np.log2(rank + 1) for rank in range(1, n_ideal + 1)))
    return dcg_at_n(recs, relevant, n) / idcg if idcg > 0 else 0.0
