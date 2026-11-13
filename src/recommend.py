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


def mmr_rerank(
    candidates: list[tuple[int, float]],
    item_matrix: sp.csr_matrix,
    lam: float,
    n: int,
) -> list[tuple[int, float]]:
    """Maximal Marginal Relevance: greedy re-ranking for diversity.

    ``candidates`` are ``(idx, relevance_score)`` pairs sorted by decreasing relevance
    (from ``cosine_scores`` + ``top_n``). At each step the candidate maximizing
    ``lam * relevance - (1 - lam) * max_sim_to_selected`` is picked (ties: lower ``idx``).
    ``lam`` = 1.0 means relevance only, 0.0 diversity only. ``item_matrix`` must be
    row-normalized (dot product = cosine). Returns up to ``n`` ``(idx, mmr_score)`` pairs.

    Vectorized: the candidate-candidate cosine matrix is computed once and the running
    ``max_sim`` is updated with the row of each newly selected item.
    """
    if not np.isfinite(lam) or not 0.0 <= lam <= 1.0:
        raise ValueError("lam must be a finite value between 0 and 1")
    if not candidates or n <= 0:
        return []

    idxs = np.array([idx for idx, _ in candidates], dtype=np.int64)
    relevance = np.array([score for _, score in candidates], dtype=np.float64)
    vecs = item_matrix[idxs]
    sims = np.asarray((vecs @ vecs.T).todense() if sp.issparse(vecs) else vecs @ vecs.T)

    max_sim = np.zeros(idxs.size)          # no item selected yet: no redundancy penalty
    available = np.ones(idxs.size, dtype=bool)
    selected: list[tuple[int, float]] = []
    for step in range(min(n, idxs.size)):
        mmr = np.where(available, lam * relevance - (1.0 - lam) * max_sim, -np.inf)
        best_value = mmr.max()
        ties = np.flatnonzero(mmr == best_value)
        best = ties[np.argmin(idxs[ties])]
        selected.append((int(idxs[best]), float(best_value)))
        available[best] = False
        max_sim = sims[best] if step == 0 else np.maximum(max_sim, sims[best])
    return selected


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
    ``mmr``: when a float (λ), applies MMR re-ranking for diversity.
    """
    scores = cosine_scores(profile, item_matrix)
    excl = np.asarray(list(exclude_idxs), dtype=np.int64)
    scores[excl] = -np.inf
    if color_identity is not None:
        if item_identity is None:
            raise ValueError("color_identity filter needs item_identity")
        scores[~color_mask(item_identity, color_identity)] = -np.inf
    if mmr is not None:
        candidates = top_n(scores, 100)
        return mmr_rerank(candidates, item_matrix, lam=float(mmr), n=n)
    return top_n(scores, n)


def recommend_for_deck(
    deck: pd.DataFrame,
    item_matrix: sp.csr_matrix,
    n: int,
    max_copies: int,
    item_identity: np.ndarray | None = None,
    color_filter: bool = False,
    mmr: Any = None,
) -> tuple[list[tuple[int, float]], np.ndarray, np.ndarray]:
    """Build the profile from the deck's train rows only and rank the catalog.

    ``deck``: rows of ``split.parquet`` for a single deck (``item_idx, copies, part``).
    ``mmr``: when a float (λ), applies MMR re-ranking for diversity.
    Returns ``(recs, profile, train_idxs)``.
    """
    train = deck[deck["part"] == "train"]
    train_idxs = train["item_idx"].to_numpy(dtype=np.int64)
    profile = build_profile(item_matrix, train_idxs, copy_weights(train["copies"], max_copies))
    identity = deck_identity(item_identity, train_idxs) if color_filter else None
    recs = recommend(
        profile, item_matrix, train_idxs, n,
        color_identity=identity, item_identity=item_identity, mmr=mmr,
    )
    return recs, profile, train_idxs
