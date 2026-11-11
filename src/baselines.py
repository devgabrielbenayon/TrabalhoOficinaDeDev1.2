"""Non-personalized baselines: random, popular and popular restricted to the deck's colors.

No leakage: popularity is counted only over the *train* cards of the *eval* decks, so a
deck's held-out test cards never contribute to the ranking. The deck's own train cards are
excluded from its list, exactly as in the content-based recommender.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from src.recommend import color_mask, top_n


def popularity_counts(split: pd.DataFrame, n_items: int, subset: str = "eval") -> np.ndarray:
    """Number of decks of ``subset`` that have each item among their train cards."""
    train = split[(split["part"] == "train") & (split["subset"] == subset)]
    counts = train.drop_duplicates(["deck_id", "item_idx"])["item_idx"].value_counts()
    pop = np.zeros(n_items, dtype=np.float64)
    pop[counts.index.to_numpy(dtype=np.int64)] = counts.to_numpy(dtype=np.float64)
    return pop


def random_recs(
    rng: np.random.Generator, n_items: int, exclude_idxs: Sequence[int] | np.ndarray, n: int
) -> list[tuple[int, float]]:
    """``n`` items drawn uniformly without replacement from the catalog minus ``exclude_idxs``."""
    candidates = np.setdiff1d(np.arange(n_items), np.asarray(list(exclude_idxs), dtype=np.int64))
    picked = rng.choice(candidates, size=min(n, candidates.size), replace=False)
    return [(int(i), 0.0) for i in picked]


def popular_recs(
    popularity: np.ndarray,
    exclude_idxs: Sequence[int] | np.ndarray,
    n: int,
    color_identity: int | None = None,
    item_identity: np.ndarray | None = None,
) -> list[tuple[int, float]]:
    """Most popular items not in ``exclude_idxs``; optionally only within ``color_identity``.

    Uses the same color filter as ``recommend.recommend`` (subset of the deck identity,
    colorless always allowed). Ties are broken by lower ``item_idx``.
    """
    scores = popularity.astype(np.float64).copy()
    scores[np.asarray(list(exclude_idxs), dtype=np.int64)] = -np.inf
    if color_identity is not None:
        if item_identity is None:
            raise ValueError("color_identity filter needs item_identity")
        scores[~color_mask(item_identity, color_identity)] = -np.inf
    return top_n(scores, n)
