"""Smoke test of the end-to-end demo on a toy catalog."""

import numpy as np
import pandas as pd
import scipy.sparse as sp

from src.demo import render_deck
from src.recommend import normalize_rows, recommend_for_deck


def _toy():
    catalog = pd.DataFrame({
        "item_idx": range(6),
        "name": ["Bolt", "Shock", "Bears", "Elf", "Counterspell", "Ornithopter"],
        "type_line": ["Instant", "Instant", "Creature — Bear", "Creature — Elf", "Instant", "Artifact Creature — Thopter"],
        "keywords": [[], [], [], [], [], ["Flying"]],
        "color_identity": [["R"], ["R"], ["G"], ["G"], ["U"], []],
    })
    X = normalize_rows(sp.csr_matrix(np.random.default_rng(0).random((6, 4))))
    names = ["text:a", "type:instant", "kw:flying", "num:cmc"]
    deck = pd.DataFrame({"deck_id": 0, "card_name": ["Bolt", "Shock", "Bears", "Elf"],
                         "item_idx": [0, 1, 2, 3], "copies": [4, 4, 2, 1],
                         "part": ["train", "train", "train", "test"], "subset": "eval"})
    cfg = {"recommend": {"max_copies": 4, "explain_top_features": 2}}
    return catalog, X, names, deck, cfg


def test_recommend_for_deck_uses_only_train() -> None:
    catalog, X, _, deck, _ = _toy()
    recs, _, train = recommend_for_deck(deck, X, 10, 4)
    assert set(train) == {0, 1, 2}
    assert {i for i, _ in recs} == {3, 4, 5}


def test_render_deck_marks_hits() -> None:
    catalog, X, names, deck, cfg = _toy()
    text = render_deck(deck, X, catalog, names, cfg, n=3, color_filter=True)
    assert "✔ Elf" in text and "Counterspell" not in text
