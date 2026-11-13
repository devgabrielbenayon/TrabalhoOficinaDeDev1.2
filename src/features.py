"""Block-wise item representation: TF-IDF text + multi-hot types/colors/keywords + numeric.

Each block is row-normalized and multiplied by its weight from ``config.yaml``.
Blocks are saved separately (unweighted) so that the weight search on the
validation decks can recombine them without recomputing anything.

No leakage: the TF-IDF vocabulary and IDF are fitted only on the rules text of
the catalog cards. That text is item metadata and is identical whether a card is
in a deck's train or test split; no deck/interaction information is used here.

Usage: python -m src.features
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from src.config import load_config
from src.preprocess import preprocess_cards

BLOCKS = ["text", "types", "colors", "keywords", "numeric"]
NUMERIC_COLUMNS = ["num_cmc", "num_power", "num_toughness", "is_creature", "variable_pt"]
COLOR_NAMES = {"W": "white", "U": "blue", "B": "black", "R": "red", "G": "green", "C": "colorless"}


# --------------------------------------------------------------------------- block builders
def type_tokens(type_line: str) -> set[str]:
    """Supertypes, card types and subtypes of every face, lower-cased."""
    tokens: set[str] = set()
    for face in type_line.split(" // "):
        left, _, right = face.partition("—")
        tokens.update(t.lower() for t in left.split())
        tokens.update(t.lower() for t in right.split())
    return tokens


def multi_hot(rows: list[set[str]], prefix: str) -> tuple[sp.csr_matrix, list[str]]:
    """Sparse multi-hot matrix with sorted, prefixed column names."""
    vocab = sorted(set().union(*rows)) if rows else []
    col = {v: j for j, v in enumerate(vocab)}
    indptr, indices = [0], []
    for r in rows:
        indices.extend(sorted(col[v] for v in r))
        indptr.append(len(indices))
    data = np.ones(len(indices), dtype=np.float64)
    mat = sp.csr_matrix((data, indices, indptr), shape=(len(rows), len(vocab)))
    return mat, [f"{prefix}:{v}" for v in vocab]


def text_block(df: pd.DataFrame, cfg: dict[str, Any]) -> tuple[sp.csr_matrix, list[str]]:
    """TF-IDF over the cleaned rules text (fitted on catalog cards only)."""
    p = cfg["features"]["tfidf"]
    vec = TfidfVectorizer(
        ngram_range=tuple(p["ngram_range"]), min_df=p["min_df"], max_features=p["max_features"],
        token_pattern=r"(?u)\b\w+\b",  # keep 1-char tokens such as damage amounts ("3")
        lowercase=False,  # already lower-cased in preprocessing
    )
    mat = vec.fit_transform(df["clean_text"])
    return mat.tocsr(), [f"text:{t}" for t in vec.get_feature_names_out()]


def build_embedding_matrix(cards: pd.DataFrame, model_name: str) -> np.ndarray:
    """Generate L2-normalized MiniLM embeddings from original card rules text.

    Unlike TF-IDF, this representation uses the original oracle_text, without
    stemming or the preprocessing pipeline.
    """
    from sentence_transformers import SentenceTransformer

    if "oracle_text" not in cards.columns:
        raise ValueError("cards must contain the 'oracle_text' column")

    texts = cards["oracle_text"].fillna("").astype(str).tolist()

    model = SentenceTransformer(model_name)
    embeddings = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    embeddings = np.asarray(embeddings, dtype=np.float32)

    if embeddings.ndim != 2 or embeddings.shape[0] != len(cards):
        raise ValueError("Invalid embedding matrix dimensions")

    if not np.isfinite(embeddings).all():
        raise ValueError("Embedding matrix contains non-finite values")

    return normalize(embeddings, norm="l2", axis=1).astype(np.float32, copy=False)


def load_or_build_embeddings(
    cards: pd.DataFrame,
    model_name: str,
    cache_path: str | Path,
) -> np.ndarray:
    """Load cached embeddings or generate and save them."""
    cache_path = Path(cache_path)
    metadata_path = cache_path.with_suffix(".json")

    texts = cards["oracle_text"].fillna("").astype(str).tolist()

    catalog_hash = hashlib.sha256(
        "\0".join(texts).encode("utf-8")
    ).hexdigest()

    expected_metadata = {
        "model_name": model_name,
        "n_cards": len(cards),
        "catalog_hash": catalog_hash,
    }

    if cache_path.exists() and metadata_path.exists():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            cached = np.load(cache_path)
            cached_norms = np.linalg.norm(cached, axis=1) if cached.ndim == 2 else np.array([])

            if (
                metadata == expected_metadata
                and cached.ndim == 2
                and cached.shape[1] > 0
                and cached.shape[0] == len(cards)
                and np.isfinite(cached).all()
                and np.all(
                    np.isclose(cached_norms, 1.0) | np.isclose(cached_norms, 0.0)
                )
            ):
                print(f"Embeddings carregados do cache: {cache_path}")
                return cached.astype(np.float32, copy=False)
            print(f"Cache de embeddings inválido ou desatualizado; regenerando: {cache_path}")
        except (ValueError, OSError, EOFError, json.JSONDecodeError) as exc:
            print(f"Não foi possível ler o cache de embeddings ({exc}); regenerando.")

    print(f"Gerando embeddings com {model_name}...")
    embeddings = build_embedding_matrix(cards, model_name)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, embeddings)
    metadata_path.write_text(
        json.dumps(expected_metadata, indent=2),
        encoding="utf-8",
    )

    print(f"Embeddings salvos em: {cache_path}")
    return embeddings



def color_block(df: pd.DataFrame) -> tuple[sp.csr_matrix, list[str]]:
    """Multi-hot of colors and of color identity (distinct prefixes); colorless is explicit."""
    rows = []
    for colors, identity in zip(df["colors"], df["color_identity"]):
        c = {f"color:{COLOR_NAMES[x]}" for x in (list(colors) or ["C"])}
        ci = {f"identity:{COLOR_NAMES[x]}" for x in (list(identity) or ["C"])}
        rows.append(c | ci)
    mat, names = multi_hot(rows, "")
    return mat, [n[1:] for n in names]  # drop the empty prefix's leading ':'


def numeric_block(df: pd.DataFrame, norm: str) -> tuple[sp.csr_matrix, list[str]]:
    """Numeric attributes (already min-max scaled to [0, 1]).

    ``norm="scaled"`` divides by sqrt(n_cols) so the block norm is <= 1 but magnitudes
    survive (a 1-drop and a 6-drop stay different). ``norm="l2"`` normalizes each row,
    which maps every non-creature to the same vector [1, 0, 0, 0, 0] regardless of cmc.
    """
    mat = sp.csr_matrix(df[NUMERIC_COLUMNS].to_numpy(dtype=np.float64))
    mat = normalize(mat) if norm == "l2" else mat / np.sqrt(len(NUMERIC_COLUMNS))
    names = ["num:cmc", "num:power", "num:toughness", "num:is_creature", "num:variable_pt"]
    return sp.csr_matrix(mat), names


def build_blocks(df: pd.DataFrame, cfg: dict[str, Any]) -> dict[str, tuple[sp.csr_matrix, list[str]]]:
    """Build every block, each L2-normalized per row (numeric: see ``numeric_block``)."""
    types, type_names = multi_hot([type_tokens(t) for t in df["type_line"]], "type")
    kws, kw_names = multi_hot([{k.lower() for k in ks} for ks in df["keywords"]], "kw")
    text, text_names = text_block(df, cfg)
    colors, color_names = color_block(df)
    blocks = {
        "text": (normalize(text), text_names),
        "types": (normalize(types), type_names),
        "colors": (normalize(colors), color_names),
        "keywords": (normalize(kws), kw_names),
        "numeric": numeric_block(df, cfg["features"].get("numeric_norm", "l2")),
    }
    return blocks


def compose(blocks: dict[str, sp.csr_matrix], weights: dict[str, float]) -> sp.csr_matrix:
    """Weighted horizontal stack of the blocks, in ``BLOCKS`` order."""
    return sp.hstack([blocks[b] * float(weights[b]) for b in BLOCKS], format="csr")


def feature_names(names: dict[str, list[str]]) -> list[str]:
    """Column names aligned with ``compose``."""
    return [n for b in BLOCKS for n in names[b]]


# --------------------------------------------------------------------------- persistence
def save_blocks(blocks: dict[str, tuple[sp.csr_matrix, list[str]]], out_dir: Path) -> None:
    """Save each unweighted block as .npz plus a JSON with column names."""
    for b, (mat, _) in blocks.items():
        sp.save_npz(out_dir / f"block_{b}.npz", mat)
    names = {b: n for b, (_, n) in blocks.items()}
    (out_dir / "feature_names.json").write_text(json.dumps(names, ensure_ascii=False), encoding="utf-8")


def load_blocks(processed: str | Path) -> tuple[dict[str, sp.csr_matrix], dict[str, list[str]]]:
    """Load the blocks saved by ``save_blocks``."""
    processed = Path(processed)
    blocks = {b: sp.load_npz(processed / f"block_{b}.npz").tocsr() for b in BLOCKS}
    names = json.loads((processed / "feature_names.json").read_text(encoding="utf-8"))
    return blocks, names


def top_features(row: sp.csr_matrix, names: list[str], k: int = 8) -> list[tuple[str, float]]:
    """Largest-valued columns of one item vector."""
    row = row.tocoo()
    order = np.argsort(-row.data)[:k]
    return [(names[row.col[i]], float(row.data[i])) for i in order]


def main() -> None:
    """Preprocess the catalog, build and save the item matrix, print a sanity sample."""
    cfg = load_config()
    processed = Path(cfg["paths"]["processed"])
    catalog = pd.read_parquet(processed / "catalog.parquet")

    df = preprocess_cards(catalog)
    df[["item_idx", "name", "clean_text", *NUMERIC_COLUMNS]].to_parquet(
        processed / "catalog_preprocessed.parquet", index=False
    )
    blocks = build_blocks(df, cfg)
    save_blocks(blocks, processed)
    mats = {b: m for b, (m, _) in blocks.items()}
    names = feature_names({b: n for b, (_, n) in blocks.items()})
    X = compose(mats, cfg["features"]["block_weights"])
    sp.save_npz(processed / "item_matrix.npz", X)

    print(f"Matriz de itens: {X.shape[0]} cartas × {X.shape[1]} features, "
          f"densidade {100 * X.nnz / np.prod(X.shape):.2f}%")
    for b in BLOCKS:
        print(f"  bloco {b:<9} {mats[b].shape[1]:>6} colunas")
    for name in cfg["features"].get("sample_cards", []):
        i = int(df.index[df["name"] == name][0])
        print(f"\n== {name}\n   texto limpo: {df.at[i, 'clean_text'][:160]}")
        for feat, val in top_features(X[i], names):
            print(f"   {val:6.3f}  {feat}")


if __name__ == "__main__":
    main()
