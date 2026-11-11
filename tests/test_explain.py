"""Tests for the explanations: they must decompose the real ranking score."""

import numpy as np
import pytest
import scipy.sparse as sp

from src.explain import contributions, explain, readable
from src.profile import build_profile
from src.recommend import normalize_rows, recommend


@pytest.fixture
def setup() -> dict:
    rng = np.random.default_rng(3)
    X = normalize_rows(sp.random(30, 12, density=0.5, random_state=3, format="csr"))
    names = [f"text:w{j}" for j in range(6)] + [f"kw:k{j}" for j in range(3)] + [f"type:t{j}" for j in range(3)]
    keywords = [{f"k{j}" for j in range(3) if rng.random() < 0.5} for _ in range(30)]
    types = [{f"t{j}" for j in range(3) if rng.random() < 0.5} for _ in range(30)]
    train = [0, 1, 2, 3]
    p = build_profile(X, train, [1.0, 0.5, 1.0, 0.25])
    return {"X": X, "names": names, "kw": keywords, "types": types, "train": train, "p": p}


def test_contributions_sum_to_cosine_score(setup) -> None:
    s = setup
    for item, score in recommend(s["p"], s["X"], s["train"], n=10):
        assert np.isclose(contributions(s["p"], s["X"][item]).sum(), score)
        exp = explain(item, s["p"], s["X"], s["train"], s["names"], s["kw"], s["types"], k=5)
        assert np.isclose(exp.score, score)


def test_top_features_are_largest_terms(setup) -> None:
    s = setup
    item, _ = recommend(s["p"], s["X"], s["train"], n=1)[0]
    exp = explain(item, s["p"], s["X"], s["train"], s["names"], s["kw"], s["types"], k=5)
    contrib = contributions(s["p"], s["X"][item])
    assert 0 < len(exp.top_features) <= 5
    assert [f.contribution for f in exp.top_features] == sorted(contrib[contrib > 0], reverse=True)[:len(exp.top_features)]
    assert sum(f.pct for f in exp.top_features) <= 100 + 1e-9


def test_most_similar_train_card(setup) -> None:
    s = setup
    item, _ = recommend(s["p"], s["X"], s["train"], n=1)[0]
    exp = explain(item, s["p"], s["X"], s["train"], s["names"], s["kw"], s["types"], k=5)
    sims = [float((s["X"][t] @ s["X"][item].T).toarray()[0, 0]) for t in s["train"]]
    assert exp.similar_item_idx in s["train"]
    assert np.isclose(exp.similar_cosine, max(sims))


def test_shared_attributes(setup) -> None:
    s = setup
    item = 10
    exp = explain(item, s["p"], s["X"], s["train"], s["names"], s["kw"], s["types"], k=5)
    for kw, n in exp.shared_keywords:
        assert kw in s["kw"][item]
        assert n == sum(kw in s["kw"][t] for t in s["train"])


def test_readable_names() -> None:
    assert readable("text:any target") == 'texto: "any target"'
    assert readable("kw:flying") == "palavra-chave: flying"
