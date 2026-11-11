"""Deck profile: weighted mean of the item vectors of the deck's train cards.

Why there is no negative feedback: a decklist only says which cards were played, never
which cards were considered and rejected. A card absent from a deck may be unknown,
unavailable, out of colors or simply worse, so absence is not a negative signal. The
profile therefore only pulls towards the cards in the deck (a Rocchio-style centroid with
the negative term dropped). This is a limitation to be discussed in the report.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import scipy.sparse as sp


def copy_weights(copies: Sequence[int] | np.ndarray, max_copies: int) -> np.ndarray:
    """Profile weight of each card: ``min(copies, max_copies) / max_copies``.

    A playset (4 copies) counts fully; a 1-of counts a quarter, since the deck relies on it less.
    """
    c = np.asarray(copies, dtype=np.float64)
    return np.minimum(c, max_copies) / float(max_copies)


def build_profile(
    item_matrix: sp.csr_matrix | np.ndarray,
    item_idxs: Sequence[int] | np.ndarray,
    weights: Sequence[float] | np.ndarray,
) -> np.ndarray:
    """L2-normalized weighted mean of the rows ``item_idxs`` of ``item_matrix``.

    Returns a dense 1-D vector with ``item_matrix.shape[1]`` entries (all zeros if the
    weighted mean is the zero vector).
    """
    idx = np.asarray(item_idxs, dtype=np.int64)
    w = np.asarray(weights, dtype=np.float64)
    if idx.shape != w.shape or idx.size == 0:
        raise ValueError("item_idxs and weights must be non-empty and of the same length")
    rows = item_matrix[idx]
    mean = np.asarray(rows.T @ w).ravel() / w.sum()
    norm = np.linalg.norm(mean)
    return mean / norm if norm > 0 else mean
