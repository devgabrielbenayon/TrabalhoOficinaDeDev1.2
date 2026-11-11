"""Tests for the evaluation pipeline."""

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from src.evaluate import (
    METHODS, best_weights, choose_decks, deck_data, evaluate_methods, metrics_markdown, per_deck_metrics,
    plot_metrics, qualitative_report, rank_cbf,
    run_methods, search_weights, weight_grid,
)
from src.features import BLOCKS, compose
from src.recommend import identity_bits, normalize_rows, recommend_for_deck

N_ITEMS = 40


@pytest.fixture
def toy() -> dict:
    rng = np.random.default_rng(5)
    blocks = {b: sp.random(N_ITEMS, 6, density=0.4, random_state=i, format="csr")
              for i, b in enumerate(BLOCKS)}
    letters = ["W", "U", "B", "R", "G"]
    identity = identity_bits([list(rng.choice(letters, size=rng.integers(0, 3), replace=False))
                              for _ in range(N_ITEMS)])
    rows = []
    for deck in range(12):
        items = rng.choice(N_ITEMS, size=8, replace=False)
        for j, item in enumerate(items):
            rows.append({"deck_id": deck, "card_name": f"c{item}", "item_idx": int(item),
                         "copies": int(rng.integers(1, 5)), "part": "test" if j < 2 else "train",
                         "subset": "validation" if deck < 4 else "eval"})
    cfg = {
        "recommend": {"max_copies": 4},
        "evaluate": {"batch_size": 5, "search_n": 5, "weight_grid": {
            "text": [1.0], "types": [0.0, 1.0], "colors": [0.5], "keywords": [0.0, 1.0], "numeric": [0.3]}},
    }
    return {"blocks": blocks, "identity": identity, "split": pd.DataFrame(rows), "cfg": cfg}


@pytest.mark.parametrize("color", [False, True])
def test_batched_ranking_matches_recommend(toy, color) -> None:
    Xn = normalize_rows(compose(toy["blocks"], dict.fromkeys(BLOCKS, 1.0)))
    data = deck_data(toy["split"], "eval", N_ITEMS, 4, toy["identity"])
    batched = rank_cbf(Xn, data, 10, batch_size=3, item_identity=toy["identity"], color_filter=color)
    for deck_id, recs in zip(data.deck_ids, batched):
        deck = toy["split"][toy["split"]["deck_id"] == deck_id]
        single, _, _ = recommend_for_deck(deck, Xn, 10, 4, toy["identity"], color)
        assert [i for i, _ in recs] == [i for i, _ in single]
        assert np.allclose([s for _, s in recs], [s for _, s in single])


def test_deck_data_subsets(toy) -> None:
    val = deck_data(toy["split"], "validation", N_ITEMS, 4, toy["identity"])
    assert list(val.deck_ids) == [0, 1, 2, 3]
    assert all(len(t) == 2 for t in val.test)
    assert all(not set(tr) & te for tr, te in zip(val.train, val.test))


def test_weight_search(toy) -> None:
    assert len(weight_grid(toy["cfg"])) == 4
    val = deck_data(toy["split"], "validation", N_ITEMS, 4, toy["identity"])
    search = search_weights(toy["blocks"], val, toy["cfg"])
    assert len(search) == 4
    assert search["ndcg@5"].is_monotonic_decreasing
    assert set(best_weights(search)) == set(BLOCKS)


def _eval_cfg(toy) -> dict:
    return toy["cfg"] | {"seed": 42, "seed_offsets": {"random_baseline": 2}}


def test_all_methods_exclude_train_and_are_reproducible(toy) -> None:
    Xn = normalize_rows(compose(toy["blocks"], dict.fromkeys(BLOCKS, 1.0)))
    data = deck_data(toy["split"], "eval", N_ITEMS, 4, toy["identity"])
    a = run_methods(Xn, data, toy["split"], toy["identity"], 10, _eval_cfg(toy))
    b = run_methods(Xn, data, toy["split"], toy["identity"], 10, _eval_cfg(toy))
    assert set(a) == set(METHODS) and a == b
    for recs in a.values():
        for r, train in zip(recs, data.train):
            assert not {i for i, _ in r} & set(train.tolist())


def test_metrics_table_and_outputs(toy, tmp_path) -> None:
    Xn = normalize_rows(compose(toy["blocks"], dict.fromkeys(BLOCKS, 1.0)))
    data = deck_data(toy["split"], "eval", N_ITEMS, 4, toy["identity"])
    recs = run_methods(Xn, data, toy["split"], toy["identity"], 10, _eval_cfg(toy))
    metrics = evaluate_methods(recs, data, Xn, [5, 10])
    assert len(metrics) == len(METHODS) * 2
    assert metrics[["precision", "recall", "ndcg", "coverage", "diversity"]].stack().between(0, 1).all()
    md = metrics_markdown(metrics, len(data.deck_ids))
    assert "## N = 5" in md and "cbf_tfidf_color" in md
    plot_metrics(metrics, 10, tmp_path / "fig.png")
    assert (tmp_path / "fig.png").stat().st_size > 0


def test_choose_decks() -> None:
    per_deck = pd.DataFrame({"deck_id": [10, 11, 12, 13, 14], "ndcg": [0.2, 0.9, 0.0, 0.5, 0.0]})
    assert choose_decks(per_deck) == [("Maior nDCG", 11), ("nDCG mediano", 10), ("nDCG = 0 (caso de erro)", 12)]
    no_zero = pd.DataFrame({"deck_id": [1, 2], "ndcg": [0.5, 0.1]})
    assert [d for _, d in choose_decks(no_zero)] == [1, 2]


def test_qualitative_report(toy) -> None:
    Xn = normalize_rows(compose(toy["blocks"], dict.fromkeys(BLOCKS, 1.0)))
    names = [f"{b}:f{j}" for b in ["text", "type", "color", "kw", "num"] for j in range(6)]
    catalog = pd.DataFrame({"item_idx": range(N_ITEMS), "name": [f"c{i}" for i in range(N_ITEMS)],
                            "keywords": [[] for _ in range(N_ITEMS)],
                            "type_line": ["Instant"] * N_ITEMS})
    cfg = _eval_cfg(toy)
    cfg["recommend"] |= {"explain_top_features": 3}
    cfg["evaluate"] |= {"qualitative_method": "cbf_tfidf", "qualitative_n": 10, "qualitative_train_preview": 3}
    data = deck_data(toy["split"], "eval", N_ITEMS, 4, toy["identity"])
    recs = run_methods(Xn, data, toy["split"], toy["identity"], 10, cfg)
    per_deck = per_deck_metrics(recs["cbf_tfidf"], data, 10)
    md = qualitative_report(toy["split"], catalog, Xn, names, per_deck, toy["identity"], cfg)
    assert md.count("**Análise:**") == len(choose_decks(per_deck))
    assert "## Maior nDCG" in md and "mais parecida no deck" in md
