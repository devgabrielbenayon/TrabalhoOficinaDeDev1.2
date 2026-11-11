"""Tests for the content-based ranking."""

import numpy as np
import pytest
import scipy.sparse as sp

from src.profile import build_profile
from src.recommend import color_mask, deck_identity, identity_bits, normalize_rows, recommend


@pytest.fixture
def X() -> sp.csr_matrix:
    rng = np.random.default_rng(0)
    return normalize_rows(sp.csr_matrix(rng.random((50, 8))))


def test_no_recommendation_in_train(X) -> None:
    train = [0, 3, 7, 11]
    p = build_profile(X, train, np.ones(len(train)))
    recs = recommend(p, X, train, n=10)
    assert len(recs) == 10
    assert not {i for i, _ in recs} & set(train)


def test_scores_are_descending(X) -> None:
    p = build_profile(X, [1, 2], [1.0, 1.0])
    scores = [s for _, s in recommend(p, X, [1, 2], n=20)]
    assert all(a >= b for a, b in zip(scores, scores[1:]))


def test_scores_are_cosines(X) -> None:
    p = build_profile(X, [5], [1.0])
    for i, s in recommend(p, X, [5], n=5):
        a, b = X[i].toarray().ravel(), X[5].toarray().ravel()
        assert np.isclose(s, a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def test_n_larger_than_candidates() -> None:
    X = normalize_rows(sp.csr_matrix(np.eye(3)))
    recs = recommend(build_profile(X, [0], [1.0]), X, [0], n=10)
    assert [i for i, _ in recs] == [1, 2]


def test_mmr_not_yet_available(X) -> None:
    with pytest.raises(NotImplementedError):
        recommend(build_profile(X, [0], [1.0]), X, [0], n=5, mmr=0.7)


def test_identity_helpers() -> None:
    ident = identity_bits([["R"], ["U", "R"], [], ["G"]])
    assert list(ident) == [8, 10, 0, 16]
    assert deck_identity(ident, [0, 2]) == 8
    assert list(color_mask(ident, 8)) == [True, False, True, False]


def test_color_filter() -> None:
    identities = [["R"], ["R"], ["U", "R"], [], ["G"], ["R", "G"], ["R"]]
    ident = identity_bits(identities)
    X = normalize_rows(sp.csr_matrix(np.random.default_rng(1).random((len(identities), 4))))
    train = [0, 1]
    p = build_profile(X, train, [1.0, 1.0])
    deck = deck_identity(ident, train)  # mono red

    unfiltered = {i for i, _ in recommend(p, X, train, n=10)}
    filtered = {i for i, _ in recommend(p, X, train, n=10, color_identity=deck, item_identity=ident)}
    assert unfiltered == {2, 3, 4, 5, 6}
    assert filtered == {3, 6}  # red and colorless only
    for i in filtered:
        assert set(identities[i]) <= {"R"}


def test_color_filter_needs_item_identity(X) -> None:
    with pytest.raises(ValueError):
        recommend(build_profile(X, [0], [1.0]), X, [0], n=5, color_identity=8)
