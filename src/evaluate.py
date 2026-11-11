"""Offline evaluation.

1. Grid search of the block weights on the *validation* decks, with mean
   nDCG@``evaluate.search_n`` as the criterion. Eval decks are never looked at there.
2. Evaluation of every method on the *eval* decks with the chosen weights, for each N in
   ``recommend.n_values``: Precision, Recall, nDCG, catalog coverage, intra-list diversity.

Scoring is batched (many decks per sparse product) but follows exactly the semantics of
``recommend.recommend_for_deck``: profile = L2-normalized copy-weighted mean of the train
rows, cosine ranking, train cards excluded, optional color-identity filter. A test checks
that both paths give the same lists.

Outputs (``results/``): weights_search.csv, metrics.csv, metrics.md, fig_metrics.png, params.json.

Usage: python -m src.evaluate [--skip-search]
"""

from __future__ import annotations

import argparse
import itertools
import json
import platform
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.preprocessing import normalize

from src.baselines import popular_recs, popularity_counts, random_recs
from src.config import load_config, make_rng
from src.features import BLOCKS, compose, load_blocks
from src.metrics import (
    catalog_coverage, intra_list_diversity, ndcg_at_n, precision_at_n, recall_at_n,
)
from src.profile import copy_weights
from src.recommend import color_mask, identity_bits, normalize_rows, top_n
from src.split import load_split

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (backend must be set first)

METHODS = ["random", "popular", "popular_color", "cbf_tfidf", "cbf_tfidf_color"]
METRICS = ["precision", "recall", "ndcg", "coverage", "diversity"]
METRIC_LABELS = {"precision": "Precision", "recall": "Recall", "ndcg": "nDCG",
                 "coverage": "Cobertura", "diversity": "Diversidade"}
LIBRARIES = ["numpy", "pandas", "scipy", "scikit-learn", "nltk", "PyYAML", "matplotlib"]
# Categorical slots 1-5 of the reference palette (light mode), one per method.
METHOD_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]

Recs = list[list[tuple[int, float]]]


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


# --------------------------------------------------------------------------- final evaluation
def run_methods(
    Xn: sp.csr_matrix,
    data: DeckData,
    split: pd.DataFrame,
    item_identity: np.ndarray,
    n: int,
    cfg: dict[str, Any],
) -> dict[str, Recs]:
    """Top-``n`` lists of every method for every deck of ``data`` (shorter N = prefixes)."""
    n_items, batch = Xn.shape[0], int(cfg["evaluate"]["batch_size"])
    pop = popularity_counts(split, n_items, subset="eval")  # train cards of eval decks only
    rng = make_rng(cfg, cfg["seed_offsets"]["random_baseline"])
    return {
        "random": [random_recs(rng, n_items, t, n) for t in data.train],
        "popular": [popular_recs(pop, t, n) for t in data.train],
        "popular_color": [popular_recs(pop, t, n, int(c), item_identity)
                          for t, c in zip(data.train, data.identity)],
        "cbf_tfidf": rank_cbf(Xn, data, n, batch),
        "cbf_tfidf_color": rank_cbf(Xn, data, n, batch, item_identity, color_filter=True),
    }


def per_deck_metrics(recs: Recs, data: DeckData, n: int) -> pd.DataFrame:
    """Precision/Recall/nDCG@n of each deck (one row per deck)."""
    rows = []
    for deck_id, r, test in zip(data.deck_ids, recs, data.test):
        ids = [i for i, _ in r]
        rows.append({"deck_id": int(deck_id), "precision": precision_at_n(ids, test, n),
                     "recall": recall_at_n(ids, test, n), "ndcg": ndcg_at_n(ids, test, n)})
    return pd.DataFrame(rows)


def evaluate_methods(
    recs_by_method: dict[str, Recs], data: DeckData, Xn: sp.csr_matrix, n_values: list[int]
) -> pd.DataFrame:
    """Mean metrics of every method for every N (one row per method and N)."""
    rows = []
    for method, recs in recs_by_method.items():
        for n in n_values:
            cut = [r[:n] for r in recs]
            ids = [[i for i, _ in r] for r in cut]
            means = per_deck_metrics(cut, data, n)[["precision", "recall", "ndcg"]].mean()
            rows.append({
                "method": method, "n": n, **{k: float(v) for k, v in means.items()},
                "coverage": catalog_coverage(ids, Xn.shape[0]),
                "diversity": float(np.mean([intra_list_diversity(i, Xn) for i in ids])),
            })
    return pd.DataFrame(rows)


def metrics_markdown(metrics: pd.DataFrame, n_decks: int) -> str:
    """Markdown table (one block per N) of the mean metrics."""
    lines = [f"# Resultados da avaliação offline\n\nMédias sobre {n_decks} decks de avaliação.\n"]
    for n, g in metrics.groupby("n"):
        lines += [f"\n## N = {n}\n",
                  "| Método | " + " | ".join(f"{METRIC_LABELS[m]}@{n}" for m in METRICS) + " |",
                  "|---|" + "---:|" * len(METRICS)]
        for _, r in g.iterrows():
            best = {m: r[m] == g[m].max() for m in METRICS}
            cells = [f"**{r[m]:.4f}**" if best[m] else f"{r[m]:.4f}" for m in METRICS]
            lines.append(f"| {r['method']} | " + " | ".join(cells) + " |")
    lines.append("\nEm negrito: melhor valor de cada coluna.")
    return "\n".join(lines) + "\n"


def plot_metrics(metrics: pd.DataFrame, n: int, path: Path) -> None:
    """Small multiples: one panel per metric, one bar per method (each panel its own scale)."""
    g = metrics[metrics["n"] == n].set_index("method").loc[METHODS]
    fig, axes = plt.subplots(1, len(METRICS), figsize=(3.2 * len(METRICS), 3.6))
    for ax, m in zip(axes, METRICS):
        bars = ax.bar(range(len(METHODS)), g[m], color=METHOD_COLORS, width=0.7,
                      edgecolor="white", linewidth=2)
        ax.bar_label(bars, fmt="%.3f", fontsize=7, color="#52514e", padding=2)
        ax.set_title(f"{METRIC_LABELS[m]}@{n}", fontsize=10, color="#0b0b0b")
        ax.set_xticks(range(len(METHODS)), METHODS, rotation=45, ha="right", fontsize=8)
        ax.set_ylim(0, max(g[m].max() * 1.2, 1e-6))
        ax.tick_params(axis="y", labelsize=7, colors="#52514e")
        ax.grid(axis="y", color="#e5e4e0", linewidth=0.6)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run_params(
    cfg: dict[str, Any], weights: dict[str, float], split: pd.DataFrame, n_items: int, n_eval: int
) -> dict[str, Any]:
    """Everything needed to reproduce the results table."""
    params = {k: v for k, v in cfg.items() if k != "paths"}
    return {
        "seed": cfg["seed"],
        "block_weights_selected": weights,
        "dataset": {"catalog_items": n_items, "decks_in_split": int(split["deck_id"].nunique()),
                    "decks_validation": int(split.loc[split["subset"] == "validation", "deck_id"].nunique()),
                    "decks_eval": n_eval, "split_rows": len(split)},
        "config": params,
        "versions": {"python": platform.python_version(), **{lib: version(lib) for lib in LIBRARIES}},
    }


# --------------------------------------------------------------------------- entry point
def main() -> None:
    """Weight search on validation, then evaluation of every method on the eval decks."""
    cfg = load_config()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-search", action="store_true",
                        help="reuse results/weights_search.csv instead of searching again")
    args = parser.parse_args()

    processed, results = Path(cfg["paths"]["processed"]), Path(cfg["paths"]["results"])
    catalog = pd.read_parquet(processed / "catalog.parquet").sort_values("item_idx")
    item_identity = identity_bits(catalog["color_identity"])
    split = load_split(processed)
    blocks, _ = load_blocks(processed)
    max_copies = int(cfg["recommend"]["max_copies"])

    search_path = results / "weights_search.csv"
    if args.skip_search and search_path.exists():
        search = pd.read_csv(search_path)
    else:
        val = deck_data(split, "validation", len(catalog), max_copies, item_identity)
        search = search_weights(blocks, val, cfg)
        search.to_csv(search_path, index=False)
        print(f"Busca de pesos: {len(search)} combinações, {len(val.deck_ids)} decks de validação")
    weights = best_weights(search)
    n_s = cfg["evaluate"]["search_n"]
    print(f"Pesos escolhidos: {weights} (nDCG@{n_s} validação = {search.at[0, f'ndcg@{n_s}']:.4f})")

    Xn = normalize_rows(compose(blocks, weights))
    data = deck_data(split, "eval", len(catalog), max_copies, item_identity)
    n_values = [int(n) for n in cfg["recommend"]["n_values"]]
    recs = run_methods(Xn, data, split, item_identity, max(n_values), cfg)
    metrics = evaluate_methods(recs, data, Xn, n_values)

    metrics.to_csv(results / "metrics.csv", index=False)
    (results / "metrics.md").write_text(metrics_markdown(metrics, len(data.deck_ids)), encoding="utf-8")
    plot_metrics(metrics, max(n_values), results / "fig_metrics.png")
    params = run_params(cfg, weights, split, len(catalog), len(data.deck_ids))
    (results / "params.json").write_text(json.dumps(params, indent=2, ensure_ascii=False), encoding="utf-8")
    print(metrics.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
