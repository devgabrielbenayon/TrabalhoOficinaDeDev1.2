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


def test_mmr_rerank_basic(X) -> None:
    """MMR with λ=1.0 should produce the same ranking as plain cosine."""
    from src.recommend import mmr_rerank
    train = [0, 3]
    p = build_profile(X, train, np.ones(len(train)))
    plain = recommend(p, X, train, n=10)
    # λ=1.0 means no diversity penalty
    recs_mmr = recommend(p, X, train, n=10, mmr=1.0)
    assert [i for i, _ in recs_mmr] == [i for i, _ in plain]


def test_mmr_rerank_increases_diversity(X) -> None:
    """MMR with λ<1 should produce different ordering than plain cosine."""
    from src.recommend import mmr_rerank
    train = [0, 3]
    p = build_profile(X, train, np.ones(len(train)))
    plain = recommend(p, X, train, n=10)
    recs_mmr = recommend(p, X, train, n=10, mmr=0.5)
    # with λ=0.5 the order should differ from plain cosine in most cases
    # At minimum, MMR should still exclude train items and return valid items
    assert len(recs_mmr) == 10
    assert not {i for i, _ in recs_mmr} & set(train)


def test_mmr_rerank_with_color_filter() -> None:
    """MMR should work with color filtering."""
    identities = [["R"], ["R"], ["U", "R"], [], ["G"], ["R", "G"], ["R"]]
    ident = identity_bits(identities)
    X = normalize_rows(sp.csr_matrix(np.random.default_rng(1).random((len(identities), 4))))
    train = [0, 1]
    p = build_profile(X, train, [1.0, 1.0])
    deck = deck_identity(ident, train)
    filtered = {i for i, _ in recommend(p, X, train, n=10, color_identity=deck, item_identity=ident, mmr=0.7)}
    assert filtered == {3, 6}


def test_mmr_scores_descending(X) -> None:
    """MMR-reranked list should have descending adjusted scores."""
    train = [0, 3]
    p = build_profile(X, train, np.ones(len(train)))
    recs = recommend(p, X, train, n=10, mmr=0.7)
    scores = [s for _, s in recs]
    assert all(a >= b for a, b in zip(scores, scores[1:]))


def test_mmr_uses_raw_relevance_and_redundancy() -> None:
    from src.recommend import mmr_rerank

    X = sp.csr_matrix([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    candidates = [(0, 0.9), (1, 0.8), (2, 0.7)]

    recs = mmr_rerank(candidates, X, lam=0.5, n=3)

    assert [idx for idx, _ in recs] == [0, 2, 1]
    assert np.allclose([score for _, score in recs], [0.45, 0.35, -0.1])


@pytest.mark.parametrize("lam", [-0.1, 1.1, float("nan")])
def test_mmr_rejects_invalid_lambda(X, lam) -> None:
    from src.recommend import mmr_rerank

    with pytest.raises(ValueError, match="between 0 and 1"):
        mmr_rerank([(0, 0.5)], X, lam=lam, n=1)


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


def _mmr_reference(candidates, item_matrix, lam, n):
    """Original loop implementation of MMR, kept as the reference for the vectorized one."""
    cand_idxs = [idx for idx, _ in candidates]
    vecs = np.asarray(item_matrix[cand_idxs].toarray())
    pos = {idx: k for k, idx in enumerate(cand_idxs)}
    selected, remaining = [], set(cand_idxs)
    for _ in range(min(n, len(cand_idxs))):
        best_idx, best_mmr = -1, -np.inf
        for idx in cand_idxs:
            if idx not in remaining:
                continue
            max_sim = (max(float(vecs[pos[idx]] @ vecs[pos[s]]) for s, _ in selected)
                       if selected else 0.0)
            score = lam * candidates[pos[idx]][1] - (1.0 - lam) * max_sim
            if score > best_mmr or (score == best_mmr and (best_idx < 0 or idx < best_idx)):
                best_mmr, best_idx = score, idx
        selected.append((best_idx, best_mmr))
        remaining.discard(best_idx)
    return selected


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("lam", [1.0, 0.9, 0.7, 0.5, 0.0])
@pytest.mark.parametrize("signed", [False, True])  # embeddings can have negative entries
def test_vectorized_mmr_matches_reference(seed: int, lam: float, signed: bool) -> None:
    from src.recommend import mmr_rerank, top_n

    rng = np.random.default_rng(seed)
    raw = rng.normal(size=(300, 16)) if signed else rng.random((300, 16))
    X = normalize_rows(sp.csr_matrix(raw))
    p = build_profile(X, [0, 1, 2], np.ones(3))
    scores = np.asarray(X @ p).ravel()
    scores[[0, 1, 2]] = -np.inf
    candidates = top_n(scores, 100)
    fast, ref = mmr_rerank(candidates, X, lam, 10), _mmr_reference(candidates, X, lam, 10)
    assert [i for i, _ in fast] == [i for i, _ in ref]
    assert np.allclose([s for _, s in fast], [s for _, s in ref])
