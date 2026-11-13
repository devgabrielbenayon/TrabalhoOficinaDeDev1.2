"""Tests for the block-wise item matrix."""

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from src.features import BLOCKS, build_blocks, compose, feature_names, numeric_block, type_tokens
from src.preprocess import preprocess_cards


@pytest.fixture
def toy_cfg() -> dict:
    return {"features": {
        "tfidf": {"ngram_range": [1, 2], "min_df": 1, "max_features": 1000},
        "block_weights": {"text": 1.0, "types": 0.5, "colors": 0.5, "keywords": 0.5, "numeric": 0.3},
        "numeric_norm": "scaled",
    }}


@pytest.fixture
def toy_cards() -> pd.DataFrame:
    df = pd.DataFrame({
        "name": ["Lightning Bolt", "Shock", "Grizzly Bears", "Wastes Golem"],
        "oracle_text": ["Lightning Bolt deals 3 damage to any target.",
                        "Shock deals 2 damage to any target.", "", "Flying"],
        "type_line": ["Instant", "Instant", "Creature — Bear", "Artifact Creature — Golem"],
        "cmc": [1.0, 1.0, 2.0, 4.0], "power": [None, None, "2", "3"], "toughness": [None, None, "2", "3"],
        "colors": [["R"], ["R"], ["G"], []], "color_identity": [["R"], ["R"], ["G"], []],
        "keywords": [[], [], [], ["Flying"]],
    })
    return preprocess_cards(df)


def test_type_tokens() -> None:
    assert type_tokens("Legendary Creature — Monkey Pirate") == {"legendary", "creature", "monkey", "pirate"}
    assert type_tokens("Creature — Human // Land") == {"creature", "human", "land"}


def test_blocks_are_unit_norm_or_empty(toy_cards: pd.DataFrame, toy_cfg: dict) -> None:
    blocks = build_blocks(toy_cards, toy_cfg)
    for b in ["text", "types", "colors", "keywords"]:
        norms = sp.linalg.norm(blocks[b][0], axis=1)
        assert np.all(np.isclose(norms, 1.0) | np.isclose(norms, 0.0)), b
    assert np.all(sp.linalg.norm(blocks["numeric"][0], axis=1) <= 1.0 + 1e-9)


def test_feature_names_aligned_and_prefixed(toy_cards: pd.DataFrame, toy_cfg: dict) -> None:
    blocks = build_blocks(toy_cards, toy_cfg)
    X = compose({b: m for b, (m, _) in blocks.items()}, toy_cfg["features"]["block_weights"])
    names = feature_names({b: n for b, (_, n) in blocks.items()})
    assert X.shape[1] == len(names)
    assert "kw:flying" in names and "type:instant" in names and "color:colorless" in names
    assert all(":" in n for n in names)


def test_block_weight_scales_block_norm(toy_cards: pd.DataFrame, toy_cfg: dict) -> None:
    blocks = {b: m for b, (m, _) in build_blocks(toy_cards, toy_cfg).items()}
    w = dict.fromkeys(BLOCKS, 0.0) | {"types": 0.5}
    X = compose(blocks, w)
    assert np.allclose(sp.linalg.norm(X, axis=1), 0.5)


def test_numeric_scaled_keeps_cmc_magnitude(toy_cards: pd.DataFrame) -> None:
    scaled, _ = numeric_block(toy_cards, "scaled")
    l2, _ = numeric_block(toy_cards, "l2")
    # Grizzly Bears (cmc 2) vs Wastes Golem (cmc 4): different in both modes...
    assert not np.allclose(scaled[2].toarray(), scaled[3].toarray())
    # ...but two non-creatures with different cmc collapse to the same vector under l2.
    two = toy_cards.iloc[[0, 1]].assign(num_cmc=[0.1, 0.9])
    l2_two, _ = numeric_block(two, "l2")
    assert np.allclose(l2_two[0].toarray(), l2_two[1].toarray())


def test_embedding_matrix_is_normalized(toy_cards, monkeypatch):
    from src import features

    class FakeModel:
        def __init__(self, model_name):
            assert model_name == "sentence-transformers/all-MiniLM-L6-v2"

        def encode(self, texts, **kwargs):
            assert len(texts) == len(toy_cards)
            assert kwargs["normalize_embeddings"] is True
            return np.array(
                [[3.0, 4.0] for _ in texts],
                dtype=np.float32,
            ) / 5.0

    monkeypatch.setattr(
        "sentence_transformers.SentenceTransformer",
        FakeModel,
    )

    embeddings = features.build_embedding_matrix(
        toy_cards,
        "sentence-transformers/all-MiniLM-L6-v2",
    )

    assert embeddings.shape == (len(toy_cards), 2)
    assert embeddings.dtype == np.float32
    assert np.allclose(np.linalg.norm(embeddings, axis=1), 1.0)


def test_embedding_cache_is_reused(toy_cards, monkeypatch, tmp_path):
    from src import features

    calls = {"count": 0}

    def fake_build(cards, model_name):
        calls["count"] += 1
        vector = np.array([1.0, 1.0, 1.0], dtype=np.float32) / np.sqrt(3)
        return np.tile(vector, (len(cards), 1)).astype(np.float32)

    monkeypatch.setattr(features, "build_embedding_matrix", fake_build)

    cache = tmp_path / "embeddings.npy"
    model = "sentence-transformers/all-MiniLM-L6-v2"

    first = features.load_or_build_embeddings(toy_cards, model, cache)
    second = features.load_or_build_embeddings(toy_cards, model, cache)

    assert calls["count"] == 1
    assert np.array_equal(first, second)
    assert np.allclose(np.linalg.norm(first, axis=1), 1.0)
    assert cache.exists()
    assert cache.with_suffix(".json").exists()
