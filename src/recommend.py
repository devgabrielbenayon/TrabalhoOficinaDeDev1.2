"""Content-based ranking: cosine between the deck profile and every catalog item.

The item matrix must be row-normalized once with ``normalize_rows`` (and the profile
built from that same normalized matrix), so the dot product equals the cosine.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import scipy.sparse as sp
from sklearn.preprocessing import normalize


def normalize_rows(item_matrix: sp.csr_matrix) -> sp.csr_matrix:
    """L2-normalize every row (zero rows stay zero)."""
    return normalize(sp.csr_matrix(item_matrix, dtype=np.float64), norm="l2", axis=1)


def cosine_scores(profile: np.ndarray, item_matrix: sp.csr_matrix) -> np.ndarray:
    """Cosine of the (unit) profile with every (unit) row of ``item_matrix``."""
    return np.asarray(item_matrix @ np.asarray(profile, dtype=np.float64)).ravel()


def top_n(scores: np.ndarray, n: int) -> list[tuple[int, float]]:
    """Indices of the ``n`` highest finite scores, descending (ties by lower index)."""
    valid = np.flatnonzero(np.isfinite(scores))
    if valid.size == 0 or n <= 0:
        return []
    order = valid[np.lexsort((valid, -scores[valid]))][:n]
    return [(int(i), float(scores[i])) for i in order]


def recommend(
    profile: np.ndarray,
    item_matrix: sp.csr_matrix,
    exclude_idxs: Sequence[int] | np.ndarray,
    n: int,
    color_identity: Any = None,
    mmr: Any = None,
) -> list[tuple[int, float]]:
    """Top-``n`` items by cosine with ``profile``, never returning ``exclude_idxs``.

    ``item_matrix`` must be row-normalized (see ``normalize_rows``).
    ``mmr`` is reserved for diversity re-ranking (Phase 6) and must be ``None`` for now.
    """
    if mmr is not None:
        raise NotImplementedError("MMR re-ranking is implemented in Phase 6")
    scores = cosine_scores(profile, item_matrix)
    excl = np.asarray(list(exclude_idxs), dtype=np.int64)
    scores[excl] = -np.inf
    return top_n(scores, n)
