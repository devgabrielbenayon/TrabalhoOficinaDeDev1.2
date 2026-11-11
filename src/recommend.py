"""Content-based ranking: cosine between the deck profile and every catalog item.

The item matrix must be row-normalized once with ``normalize_rows`` (and the profile
built from that same normalized matrix), so the dot product equals the cosine.
"""

from __future__ import annotations

from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.preprocessing import normalize

from src.profile import build_profile, copy_weights

COLOR_BITS = {"W": 1, "U": 2, "B": 4, "R": 8, "G": 16}


# --------------------------------------------------------------------------- color identity
def identity_bits(identities: Iterable[Iterable[str]]) -> np.ndarray:
    """Encode each card's color identity (e.g. ``["U", "R"]``) as a 5-bit mask; colorless = 0."""
    return np.array([sum(COLOR_BITS[c] for c in set(ci)) for ci in identities], dtype=np.int64)


def deck_identity(item_identity: np.ndarray, train_idxs: Sequence[int] | np.ndarray) -> int:
    """Deck identity = union (bitwise OR) of the identities of its train cards."""
    idx = np.asarray(list(train_idxs), dtype=np.int64)
    return int(np.bitwise_or.reduce(item_identity[idx])) if idx.size else 0


def color_mask(item_identity: np.ndarray, deck_bits: int) -> np.ndarray:
    """True where the card's identity is a subset of the deck's (colorless always passes)."""
    return (item_identity & ~np.int64(deck_bits)) == 0


# --------------------------------------------------------------------------- ranking
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
    color_identity: int | None = None,
    mmr: Any = None,
    item_identity: np.ndarray | None = None,
) -> list[tuple[int, float]]:
    """Top-``n`` items by cosine with ``profile``, never returning ``exclude_idxs``.

    ``item_matrix`` must be row-normalized (see ``normalize_rows``).
    ``color_identity``: optional deck identity bits (see ``deck_identity``); when given, only
    cards whose identity is a subset of it are ranked. Requires ``item_identity``
    (``identity_bits`` of the catalog, aligned with the matrix rows).
    ``mmr`` is reserved for diversity re-ranking (Phase 6) and must be ``None`` for now.
    """
    if mmr is not None:
        raise NotImplementedError("MMR re-ranking is implemented in Phase 6")
    scores = cosine_scores(profile, item_matrix)
    excl = np.asarray(list(exclude_idxs), dtype=np.int64)
    scores[excl] = -np.inf
    if color_identity is not None:
        if item_identity is None:
            raise ValueError("color_identity filter needs item_identity")
        scores[~color_mask(item_identity, color_identity)] = -np.inf
    return top_n(scores, n)


def recommend_for_deck(
    deck: pd.DataFrame,
    item_matrix: sp.csr_matrix,
    n: int,
    max_copies: int,
    item_identity: np.ndarray | None = None,
    color_filter: bool = False,
) -> tuple[list[tuple[int, float]], np.ndarray, np.ndarray]:
    """Build the profile from the deck's train rows only and rank the catalog.

    ``deck``: rows of ``split.parquet`` for a single deck (``item_idx, copies, part``).
    Returns ``(recs, profile, train_idxs)``.
    """
    train = deck[deck["part"] == "train"]
    train_idxs = train["item_idx"].to_numpy(dtype=np.int64)
    profile = build_profile(item_matrix, train_idxs, copy_weights(train["copies"], max_copies))
    identity = deck_identity(item_identity, train_idxs) if color_filter else None
    recs = recommend(profile, item_matrix, train_idxs, n, color_identity=identity, item_identity=item_identity)
    return recs, profile, train_idxs
