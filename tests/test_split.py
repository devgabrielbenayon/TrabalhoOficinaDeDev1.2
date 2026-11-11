"""Tests for the per-deck train/test split."""

import numpy as np
import pandas as pd
import pytest

from src.split import n_test_cards, split_decks


@pytest.fixture
def toy_cfg() -> dict:
    return {
        "seed": 42,
        "seed_offsets": {"split": 1},
        "split": {"test_fraction": 0.2, "validation_fraction": 0.1, "min_cards_per_deck": 10},
    }


@pytest.fixture
def toy_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(0)
    names = [f"Card {i:03d}" for i in range(200)]
    catalog = pd.DataFrame({"item_idx": range(len(names)), "name": names})
    rows = []
    for deck_id in range(60):
        size = 5 if deck_id < 3 else int(rng.integers(10, 40))  # 3 decks too small
        for name in rng.choice(names, size=size, replace=False):
            rows.append({"deck_id": deck_id, "date": "2025-01-01", "card_name": name,
                         "copies": int(rng.integers(1, 5))})
    return pd.DataFrame(rows), catalog


def test_train_and_test_are_disjoint(toy_data, toy_cfg) -> None:
    split, _ = split_decks(*toy_data, toy_cfg)
    for _, g in split.groupby("deck_id"):
        train = set(g.loc[g["part"] == "train", "item_idx"])
        test = set(g.loc[g["part"] == "test", "item_idx"])
        assert train and test and not train & test


def test_same_seed_same_split(toy_data, toy_cfg) -> None:
    a, _ = split_decks(*toy_data, toy_cfg)
    b, _ = split_decks(*toy_data, toy_cfg)
    pd.testing.assert_frame_equal(a, b)
    c, _ = split_decks(*toy_data, toy_cfg | {"seed": 7})
    assert not a["part"].equals(c["part"])


def test_test_fraction_close_to_target(toy_data, toy_cfg) -> None:
    split, report = split_decks(*toy_data, toy_cfg)
    assert abs(report["test_fraction_observed"] - 0.2) < 0.03
    per_deck = split.groupby("deck_id")["part"].apply(lambda s: (s == "test").mean())
    assert per_deck.between(0.1, 0.3).all()


def test_every_deck_has_a_test_card(toy_data, toy_cfg) -> None:
    split, _ = split_decks(*toy_data, toy_cfg)
    assert (split.groupby("deck_id")["part"].apply(lambda s: (s == "test").sum()) >= 1).all()
    assert n_test_cards(2, 0.2) == 1 and n_test_cards(10, 0.2) == 2


def test_small_decks_dropped_and_reported(toy_data, toy_cfg) -> None:
    split, report = split_decks(*toy_data, toy_cfg)
    assert report["decks_dropped_small"] == 3
    assert not split["deck_id"].isin([0, 1, 2]).any()
    assert report["decks_validation"] == round(0.1 * report["decks_kept"])
    # A deck is entirely validation or entirely eval.
    assert (split.groupby("deck_id")["subset"].nunique() == 1).all()
