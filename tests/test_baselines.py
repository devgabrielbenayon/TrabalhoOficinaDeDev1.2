"""Tests for the baselines."""

import numpy as np
import pandas as pd

from src.baselines import popular_recs, popularity_counts, random_recs
from src.recommend import identity_bits


def _split() -> pd.DataFrame:
    rows = [
        # deck, item, part, subset
        (0, 0, "train", "eval"), (0, 1, "train", "eval"), (0, 2, "test", "eval"),
        (1, 0, "train", "eval"), (1, 3, "train", "eval"), (1, 2, "test", "eval"),
        (2, 4, "train", "validation"), (2, 2, "test", "validation"),
    ]
    return pd.DataFrame(rows, columns=["deck_id", "item_idx", "part", "subset"])


def test_popularity_uses_only_eval_train_cards() -> None:
    pop = popularity_counts(_split(), n_items=5)
    assert list(pop) == [2, 1, 0, 1, 0]  # item 2 is test-only, item 4 is validation-only


def test_popular_excludes_train_and_ranks_by_count() -> None:
    pop = np.array([5.0, 3.0, 3.0, 9.0, 1.0])
    recs = popular_recs(pop, exclude_idxs=[3], n=3)
    assert [i for i, _ in recs] == [0, 1, 2]


def test_popular_color_filter() -> None:
    ident = identity_bits([["R"], ["G"], [], ["R"], ["U"]])
    pop = np.array([1.0, 9.0, 2.0, 3.0, 8.0])
    recs = popular_recs(pop, [0], 5, color_identity=8, item_identity=ident)
    assert [i for i, _ in recs] == [3, 2]


def test_random_is_reproducible_and_excludes_train() -> None:
    a = random_recs(np.random.default_rng(1), 100, [0, 1, 2], 10)
    b = random_recs(np.random.default_rng(1), 100, [0, 1, 2], 10)
    assert a == b
    ids = [i for i, _ in a]
    assert len(set(ids)) == 10 and not set(ids) & {0, 1, 2}
