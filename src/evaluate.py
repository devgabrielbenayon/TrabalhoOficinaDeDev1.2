"""Offline evaluation.

Part 1 (this file so far): grid search of the block weights on the *validation* decks,
with mean nDCG@``evaluate.search_n`` as the criterion. Eval decks are never looked at here.

Scoring is batched (many decks per sparse product) but follows exactly the semantics of
``recommend.recommend_for_deck``: profile = L2-normalized copy-weighted mean of the train
rows, cosine ranking, train cards excluded, optional color-identity filter. A test checks
that both paths give the same lists.

Usage: python -m src.evaluate
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.preprocessing import normalize

from src.config import load_config
from src.features import BLOCKS, compose, load_blocks
from src.metrics import ndcg_at_n
from src.profile import copy_weights
from src.recommend import color_mask, identity_bits, normalize_rows, top_n
from src.split import load_split


# --------------------------------------------------------------------------- deck data
@dataclass
class DeckData:
    """Train/test cards of a set of decks, in matrix form."""

    deck_ids: np.ndarray        # (n_decks,)
    weights: sp.csr_matrix      # (n_decks, n_items): copy weight of each train card, 0 elsewhere
    train: list[np.ndarray]     # train item_idxs per deck
    test: list[set[int]]        # test item_idxs per deck
    identity: np.ndarray        # (n_decks,): color identity bits (union of train cards)


def deck_data(
    split: pd.DataFrame, subset: str, n_items: int, max_copies: int, item_identity: np.ndarray
) -> DeckData:
    """Collect the decks of ``subset`` (``"validation"`` or ``"eval"``) from the split."""
    df = split[split["subset"] == subset].sort_values(["deck_id", "item_idx"])
    deck_ids = np.sort(df["deck_id"].unique())
    row_of = {d: r for r, d in enumerate(deck_ids)}

    tr = df[df["part"] == "train"]
    rows = tr["deck_id"].map(row_of).to_numpy()
    cols = tr["item_idx"].to_numpy(dtype=np.int64)
    W = sp.csr_matrix((copy_weights(tr["copies"], max_copies), (rows, cols)),
                      shape=(len(deck_ids), n_items))

    train = [W.indices[W.indptr[r]:W.indptr[r + 1]].astype(np.int64) for r in range(len(deck_ids))]
    test_groups = df[df["part"] == "test"].groupby("deck_id")["item_idx"].apply(set)
    test = [test_groups.get(d, set()) for d in deck_ids]
    identity = np.array([int(np.bitwise_or.reduce(item_identity[t])) if t.size else 0 for t in train],
                        dtype=np.int64)
    return DeckData(deck_ids, W, train, test, identity)


# --------------------------------------------------------------------------- batched CBF
def rank_cbf(
    Xn: sp.csr_matrix,
    data: DeckData,
    n: int,
    batch_size: int,
    item_identity: np.ndarray | None = None,
    color_filter: bool = False,
) -> list[list[tuple[int, float]]]:
    """Top-``n`` content-based lists for every deck (``Xn``: row-normalized item matrix)."""
    profiles = normalize(data.weights @ Xn)  # mean scale is irrelevant after L2 normalization
    out: list[list[tuple[int, float]]] = []
    for start in range(0, len(data.deck_ids), batch_size):
        stop = min(start + batch_size, len(data.deck_ids))
        scores = np.asarray((profiles[start:stop] @ Xn.T).todense(), dtype=np.float64)
        for r in range(stop - start):
            s = scores[r]
            s[data.train[start + r]] = -np.inf
            if color_filter:
                s[~color_mask(item_identity, int(data.identity[start + r]))] = -np.inf
            out.append(top_n(s, n))
    return out


def mean_ndcg(recs: list[list[tuple[int, float]]], data: DeckData, n: int) -> float:
    """Mean nDCG@n over the decks."""
    return float(np.mean([ndcg_at_n([i for i, _ in r], t, n) for r, t in zip(recs, data.test)]))


# --------------------------------------------------------------------------- weight search
def weight_grid(cfg: dict[str, Any]) -> list[dict[str, float]]:
    """Every combination of the per-block values in ``evaluate.weight_grid``."""
    grid = cfg["evaluate"]["weight_grid"]
    combos = itertools.product(*(grid[b] for b in BLOCKS))
    return [dict(zip(BLOCKS, map(float, c))) for c in combos]


def search_weights(
    blocks: dict[str, sp.csr_matrix], val: DeckData, cfg: dict[str, Any]
) -> pd.DataFrame:
    """Mean validation nDCG@search_n of every weight combination, best first."""
    ev = cfg["evaluate"]
    n, batch = int(ev["search_n"]), int(ev["batch_size"])
    rows = []
    for w in weight_grid(cfg):
        if not any(w.values()):
            continue
        Xn = normalize_rows(compose(blocks, w))
        rows.append(w | {f"ndcg@{n}": mean_ndcg(rank_cbf(Xn, val, n, batch), val, n)})
    df = pd.DataFrame(rows)
    return df.sort_values([f"ndcg@{n}", *BLOCKS], ascending=[False] + [True] * len(BLOCKS),
                          ignore_index=True)


def best_weights(search: pd.DataFrame) -> dict[str, float]:
    """Weights of the first (best) row of a ``search_weights`` table."""
    return {b: float(search.at[0, b]) for b in BLOCKS}


def main() -> None:
    """Run the validation weight search and save ``results/weights_search.csv``."""
    cfg = load_config()
    processed, results = Path(cfg["paths"]["processed"]), Path(cfg["paths"]["results"])
    catalog = pd.read_parquet(processed / "catalog.parquet").sort_values("item_idx")
    item_identity = identity_bits(catalog["color_identity"])
    split = load_split(processed)
    blocks, _ = load_blocks(processed)
    val = deck_data(split, "validation", len(catalog), int(cfg["recommend"]["max_copies"]), item_identity)

    search = search_weights(blocks, val, cfg)
    search.to_csv(results / "weights_search.csv", index=False)
    n = cfg["evaluate"]["search_n"]
    print(f"Busca de pesos: {len(search)} combinações, {len(val.deck_ids)} decks de validação")
    print(search.head(5).to_string(index=False))
    print(f"Melhor: {best_weights(search)} (nDCG@{n} = {search.at[0, f'ndcg@{n}']:.4f})")


if __name__ == "__main__":
    main()
