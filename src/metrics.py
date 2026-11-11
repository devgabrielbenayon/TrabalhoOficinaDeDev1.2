"""Offline ranking metrics with binary relevance (a card is relevant iff it is in the deck's test set).

``recs`` is always a ranked list of item indices (best first); ``relevant`` a set of item indices.
"""

from __future__ import annotations

from typing import Collection, Iterable, Sequence

import numpy as np
import scipy.sparse as sp
from sklearn.preprocessing import normalize


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


def catalog_coverage(all_recs: Iterable[Sequence[int]], n_items: int) -> float:
    """Fraction of the catalog that appears in at least one recommendation list."""
    seen = {i for recs in all_recs for i in recs}
    return len(seen) / n_items if n_items else 0.0


def intra_list_diversity(recs: Sequence[int], item_matrix: sp.csr_matrix) -> float:
    """``1 - mean pairwise cosine`` of the items in one list (0 for lists with < 2 items).

    Rows are re-normalized here, so any item matrix (weighted or not) can be passed.
    """
    idx = np.asarray(list(recs), dtype=np.int64)
    if idx.size < 2:
        return 0.0
    rows = normalize(sp.csr_matrix(item_matrix[idx], dtype=np.float64))
    sims = (rows @ rows.T).toarray()
    upper = sims[np.triu_indices(idx.size, k=1)]
    return float(1.0 - upper.mean())
