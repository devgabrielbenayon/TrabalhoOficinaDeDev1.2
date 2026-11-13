"""Offline evaluation.

1. Grid search of the block weights on the *validation* decks, with mean
   nDCG@``evaluate.search_n`` as the criterion. Eval decks are never looked at there.
2. Evaluation of every method on the *eval* decks with the chosen weights, for each N in
   ``recommend.n_values``: Precision, Recall, nDCG, catalog coverage, intra-list diversity.

Scoring is batched (many decks per sparse product) but follows exactly the semantics of
``recommend.recommend_for_deck``: profile = L2-normalized copy-weighted mean of the train
rows, cosine ranking, train cards excluded, optional color-identity filter. A test checks
that both paths give the same lists.

3. Qualitative report: best, median and a zero-nDCG eval deck with explained Top-N.

Outputs (``results/``): weight/lambda search CSVs, metrics.csv, metrics.md,
fig_metrics.png, fig_mmr_tradeoff.png, params.json and qualitative.md.

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
from typing import Any, Mapping

import matplotlib
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.preprocessing import normalize
from tqdm import tqdm

from src.baselines import popular_recs, popularity_counts, random_recs
from src.config import load_config, make_rng
from src.explain import explain, item_attributes
from src.features import BLOCKS, compose, feature_names, load_blocks, load_or_build_embeddings
from src.metrics import (
    catalog_coverage, intra_list_diversity, ndcg_at_n, precision_at_n, recall_at_n,
)
from src.profile import copy_weights
from src.recommend import (
    COLOR_BITS, color_mask, identity_bits, mmr_rerank, normalize_rows, recommend_for_deck, top_n,
)
from src.split import load_split

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (backend must be set first)

METHODS = [
    "random",
    "popular",
    "popular_color",
    "cbf_tfidf",
    "cbf_tfidf_color",
    "cbf_tfidf_mmr",
    "cbf_emb",
]
METRICS = ["precision", "recall", "ndcg", "coverage", "diversity"]
METRIC_LABELS = {"precision": "Precision", "recall": "Recall", "ndcg": "nDCG",
                 "coverage": "Cobertura", "diversity": "Diversidade"}
LIBRARIES = ["numpy", "pandas", "scipy", "scikit-learn", "nltk", "PyYAML", "matplotlib",
             "torch", "sentence-transformers", "streamlit"]
# Categorical slots 1-5 of the reference palette (light mode), one per method.
METHOD_COLORS = [
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#8b6bd6", "#6a9c35",
]

Recs = list[list[tuple[int, float]]]
DENSE_THRESHOLD = 0.05  # item matrices denser than this are scored with dense algebra


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
    # Dense embedding blocks make sparse products slow: use dense algebra when the matrix is dense.
    dense = Xn.nnz > DENSE_THRESHOLD * Xn.shape[0] * Xn.shape[1]
    if dense:
        profiles, XnT = profiles.toarray(), Xn.toarray().T
    out: list[list[tuple[int, float]]] = []
    for start in range(0, len(data.deck_ids), batch_size):
        stop = min(start + batch_size, len(data.deck_ids))
        if dense:
            scores = profiles[start:stop] @ XnT
        else:
            scores = np.asarray((profiles[start:stop] @ Xn.T).todense(), dtype=np.float64)
        for r in range(stop - start):
            s = scores[r]
            s[data.train[start + r]] = -np.inf
            if color_filter:
                s[~color_mask(item_identity, int(data.identity[start + r]))] = -np.inf
            out.append(top_n(s, n))
    return out


def rank_cbf_mmr(
    Xn: sp.csr_matrix,
    data: DeckData,
    n: int,
    batch_size: int,
    lam: float,
) -> Recs:
    """Re-rank each deck's 100 highest-cosine TF-IDF candidates with MMR."""
    candidates = rank_cbf(Xn, data, 100, batch_size)
    return [mmr_rerank(row, Xn, lam, n) for row in candidates]


def mean_ndcg(recs: list[list[tuple[int, float]]], data: DeckData, n: int) -> float:
    """Mean nDCG@n over the decks."""
    return float(np.mean([ndcg_at_n([i for i, _ in r], t, n) for r, t in zip(recs, data.test)]))


def search_mmr_lambda(
    Xn: sp.csr_matrix,
    validation: DeckData,
    lambdas: list[float],
    n: int,
    batch_size: int,
) -> pd.DataFrame:
    """Choose λ by validation nDCG@n; tie-break by diversity, then larger λ."""
    candidates = rank_cbf(Xn, validation, 100, batch_size)
    rows = []
    for lam in lambdas:
        recs = [mmr_rerank(row, Xn, float(lam), n) for row in candidates]
        rows.append({
            "lambda": float(lam),
            f"ndcg@{n}": mean_ndcg(recs, validation, n),
            "diversity": float(np.mean([
                intra_list_diversity([i for i, _ in rec], Xn) for rec in recs
            ])),
        })
    return pd.DataFrame(rows).sort_values(
        [f"ndcg@{n}", "diversity", "lambda"],
        ascending=[False, False, False],
        ignore_index=True,
    )


def best_mmr_lambda(search: pd.DataFrame) -> float:
    """Return the λ selected by validation nDCG (with deterministic tie-breaks)."""
    if search.empty:
        raise ValueError("MMR lambda search produced no candidates")
    return float(search.at[0, "lambda"])


# --------------------------------------------------------------------------- weight search
def weight_grid(cfg: dict[str, Any]) -> list[dict[str, float]]:
    """Every combination of the per-block values in ``evaluate.weight_grid``."""
    grid = cfg["evaluate"]["weight_grid"]
    combos = itertools.product(*(grid[b] for b in BLOCKS))
    return [dict(zip(BLOCKS, map(float, c))) for c in combos]


def search_weights(
    blocks: dict[str, sp.csr_matrix], val: DeckData, cfg: dict[str, Any], desc: str = "busca de pesos"
) -> pd.DataFrame:
    """Mean validation nDCG@search_n of every weight combination, best first."""
    ev = cfg["evaluate"]
    n, batch = int(ev["search_n"]), int(ev["batch_size"])
    rows = []
    for w in tqdm(weight_grid(cfg), desc=desc, unit="comb", mininterval=5):
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
    Xn_emb: sp.csr_matrix | None = None,
    mmr_lambda: float = 1.0,
) -> dict[str, Recs]:
    """Top-n lists for all methods. Xn_emb is the MiniLM representation."""
    n_items, batch = Xn.shape[0], int(cfg["evaluate"]["batch_size"])

    # Without embeddings (tests, old callers) cbf_emb falls back to the TF-IDF matrix.
    if Xn_emb is None:
        Xn_emb = Xn

    if Xn_emb.shape[0] != n_items:
        raise ValueError("TF-IDF and embedding matrices must have the same number of items")

    pop = popularity_counts(split, n_items, subset="eval")
    rng = make_rng(cfg, cfg["seed_offsets"]["random_baseline"])

    return {
        "random": [random_recs(rng, n_items, t, n) for t in data.train],
        "popular": [popular_recs(pop, t, n) for t in data.train],
        "popular_color": [
            popular_recs(pop, t, n, int(c), item_identity)
            for t, c in zip(data.train, data.identity)
        ],
        "cbf_tfidf": rank_cbf(Xn, data, n, batch),
        "cbf_tfidf_color": rank_cbf(
            Xn, data, n, batch, item_identity, color_filter=True
        ),
        "cbf_tfidf_mmr": rank_cbf_mmr(Xn, data, n, batch, mmr_lambda),
        "cbf_emb": rank_cbf(Xn_emb, data, n, batch),
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
    recs_by_method: dict[str, Recs],
    data: DeckData,
    Xn: sp.csr_matrix,
    n_values: list[int],
    matrices_by_method: dict[str, sp.csr_matrix] | None = None,
) -> pd.DataFrame:
    """Mean metrics of each method and N.

    Diversity uses ``Xn`` for every method (one common space) unless ``matrices_by_method``
    overrides it for a method.
    """
    rows = []

    for method, recs in recs_by_method.items():
        metric_Xn = (
            matrices_by_method.get(method, Xn)
            if matrices_by_method is not None
            else Xn
        )

        for n in n_values:
            cut = [r[:n] for r in recs]
            ids = [[i for i, _ in r] for r in cut]

            means = per_deck_metrics(cut, data, n)[
                ["precision", "recall", "ndcg"]
            ].mean()

            rows.append({
                "method": method,
                "n": n,
                **{k: float(v) for k, v in means.items()},
                "coverage": catalog_coverage(ids, Xn.shape[0]),
                "diversity": float(np.mean([
                    intra_list_diversity(items, metric_Xn)
                    for items in ids
                ])),
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


def plot_mmr_tradeoff(
    tradeoff: pd.DataFrame,
    selected_lambda: float,
    path: Path,
) -> None:
    """Plot eval nDCG@10 against diversity across the validation-selected λ grid."""
    points = tradeoff.sort_values("lambda")
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(points["diversity"], points["ndcg@10"], color="#2a78d6", linewidth=1.5)
    ax.scatter(points["diversity"], points["ndcg@10"], color="#2a78d6", zorder=2)
    for _, row in points.iterrows():
        ax.annotate(
            f"λ={row['lambda']:.1f}",
            (row["diversity"], row["ndcg@10"]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
        )
    selected = points[np.isclose(points["lambda"], selected_lambda)]
    if not selected.empty:
        ax.scatter(
            selected["diversity"], selected["ndcg@10"],
            s=90, facecolors="none", edgecolors="#eb6834", linewidths=2,
            label=f"λ selecionado na validação = {selected_lambda:.1f}",
            zorder=3,
        )
        ax.legend()
    ax.set_xlabel("Diversidade intra-lista")
    ax.set_ylabel("nDCG@10")
    ax.set_title("Trade-off de MMR (conjunto de avaliação)")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run_params(
    cfg: dict[str, Any],
    weights: dict[str, float],
    split: pd.DataFrame,
    n_items: int,
    n_eval: int,
    mmr_lambda: float,
    embedding_weights: dict[str, float],
) -> dict[str, Any]:
    """Everything needed to reproduce the results table."""
    params = {k: v for k, v in cfg.items() if k != "paths"}
    return {
        "seed": cfg["seed"],
        "block_weights_selected": weights,
        "embedding_block_weights_selected": embedding_weights,
        "mmr_lambda_selected": mmr_lambda,
        "dataset": {"catalog_items": n_items, "decks_in_split": int(split["deck_id"].nunique()),
                    "decks_validation": int(split.loc[split["subset"] == "validation", "deck_id"].nunique()),
                    "decks_eval": n_eval, "split_rows": len(split)},
        "config": params,
        "versions": {"python": platform.python_version(), **{lib: version(lib) for lib in LIBRARIES}},
    }


# --------------------------------------------------------------------------- qualitative report
def _colors(bits: int) -> str:
    """Color identity bits -> 'WUBRG'-style string."""
    return "".join(c for c, b in COLOR_BITS.items() if bits & b) or "incolor"


def _pick_new_identity(
    candidates: list[int], start: int, identity: Mapping[int, int] | None, used: set[int]
) -> int:
    """Deck closest to position ``start`` whose color identity was not used yet (else ``start``)."""
    if identity is not None:
        order = sorted(range(len(candidates)), key=lambda k: (abs(k - start), k))
        for k in order:
            if identity.get(candidates[k]) not in used:
                return candidates[k]
    return candidates[start]


def choose_decks(
    per_deck: pd.DataFrame, identity: Mapping[int, int] | None = None
) -> list[tuple[str, int]]:
    """Best deck, a median deck among those with at least one hit, and a zero-nDCG deck.

    The median is taken over decks with nDCG > 0: more than half of the decks may have
    nDCG = 0, and then the plain median would be a second error case. When ``identity``
    (deck_id -> color identity bits) is given, the median and error decks are moved to the
    nearest deck whose color identity differs from the decks already chosen, so the
    report shows different archetypes. Ties are broken by lowest deck_id.
    """
    ranked = per_deck.sort_values(["ndcg", "deck_id"], ascending=[False, True], ignore_index=True)
    best = int(ranked.at[0, "deck_id"])
    chosen = [("Maior nDCG", best)]
    used = {identity.get(best)} if identity is not None else set()

    hits = per_deck[(per_deck["ndcg"] > 0) & (per_deck["deck_id"] != best)]
    hits = hits.sort_values(["ndcg", "deck_id"])["deck_id"].astype(int).tolist()
    if hits:
        median = _pick_new_identity(hits, len(hits) // 2, identity, used)
        chosen.append(("nDCG mediano entre os decks com acerto", median))
        used.add(identity.get(median) if identity is not None else None)

    zeros = sorted(per_deck.loc[per_deck["ndcg"] == 0, "deck_id"].astype(int))
    if zeros:
        chosen.append(("nDCG = 0 (caso de erro)", _pick_new_identity(zeros, 0, identity, used)))
    return chosen


def qualitative_report(
    split: pd.DataFrame,
    catalog: pd.DataFrame,
    Xn: sp.csr_matrix,
    names: list[str],
    per_deck: pd.DataFrame,
    item_identity: np.ndarray,
    cfg: dict[str, Any],
) -> str:
    """Markdown with train summary, test, explained Top-N and hits for the chosen decks."""
    ev, rc = cfg["evaluate"], cfg["recommend"]
    method, n = ev["qualitative_method"], int(ev["qualitative_n"])
    card_names = catalog.sort_values("item_idx")["name"].tolist()
    keywords, types = item_attributes(catalog)
    lines = [f"# Análise qualitativa ({method}, Top-{n})\n",
             "Gerado por `python -m src.evaluate`. ✔ marca acerto (carta do teste do deck).\n"]
    ndcg = per_deck.set_index("deck_id")["ndcg"]
    train_rows = split[split["part"] == "train"]
    deck_ids = set(per_deck["deck_id"].astype(int))
    identity = {
        int(d): int(np.bitwise_or.reduce(item_identity[g["item_idx"].to_numpy()]))
        for d, g in train_rows[train_rows["deck_id"].isin(deck_ids)].groupby("deck_id")
    }
    for label, deck_id in choose_decks(per_deck, identity):
        deck = split[split["deck_id"] == deck_id]
        recs, profile, train_idxs = recommend_for_deck(
            deck, Xn, n, int(rc["max_copies"]), item_identity, color_filter=method.endswith("_color"))
        train = deck[deck["part"] == "train"].sort_values(["copies", "card_name"], ascending=[False, True])
        test = set(deck.loc[deck["part"] == "test", "item_idx"])
        preview = int(ev["qualitative_train_preview"])
        shown = ", ".join(f"{r.copies}x {r.card_name}" for r in train.head(preview).itertuples())
        extra = f" … (+{len(train) - preview})" if len(train) > preview else ""
        lines += [
            f"\n## {label}: deck {deck_id}, cores {_colors(identity.get(deck_id, 0))} "
            f"(nDCG@{n} = {ndcg[deck_id]:.3f})\n",
            f"**Treino ({len(train)} cartas):** {shown}{extra}\n",
            f"**Teste ({len(test)} cartas):** " + ", ".join(sorted(card_names[i] for i in test)) + "\n",
            f"**Top-{n}** ({sum(i in test for i, _ in recs)} acertos):\n",
        ]
        for rank, (i, score) in enumerate(recs, 1):
            exp = explain(i, profile, Xn, train_idxs, names, keywords, types, int(rc["explain_top_features"]))
            feats = "; ".join(f"{f.label} ({f.pct:.0f}%)" for f in exp.top_features)
            kws = ", ".join(a for a, _ in exp.shared_keywords) or "nenhuma"
            tps = ", ".join(a for a, _ in exp.shared_types) or "nenhum"
            lines += [
                f"{rank}. {'✔ ' if i in test else ''}**{card_names[i]}** (score {score:.3f})",
                f"   - features: {feats}",
                f"   - mais parecida no deck: {card_names[exp.similar_item_idx]} "
                f"(cosseno {exp.similar_cosine:.2f})",
                f"   - em comum com o deck: palavras-chave {kws}; tipos {tps}",
            ]
        lines.append("\n**Análise:** _(a preencher pela equipe)_\n")
    return "\n".join(lines)


# --------------------------------------------------------------------------- entry point
def main() -> None:
    """Weight search on validation, then evaluation of every method on the eval decks."""
    cfg = load_config()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-search",
        action="store_true",
        help="reuse the saved TF-IDF, MiniLM and MMR validation searches",
    )
    args = parser.parse_args()

    processed, results = Path(cfg["paths"]["processed"]), Path(cfg["paths"]["results"])
    catalog = pd.read_parquet(processed / "catalog.parquet").sort_values("item_idx")
    item_identity = identity_bits(catalog["color_identity"])
    split = load_split(processed)
    blocks, block_names = load_blocks(processed)

    # Load (or build and cache) the MiniLM embeddings.
    embedding_cfg = cfg["features"]["embeddings"]

    if not embedding_cfg.get("enabled", False):
        raise ValueError(
            "A avaliação cbf_emb exige embeddings habilitados no config.yaml"
        )

    embeddings = load_or_build_embeddings(
        catalog,
        embedding_cfg["model"],
        processed / "embeddings.npy",
    )

    if embeddings.shape[0] != len(catalog):
        raise ValueError(
            "A quantidade de embeddings não corresponde ao catálogo"
        )

    # Embeddings replace only the text block; the other blocks stay.
    embedding_blocks = dict(blocks)
    embedding_blocks["text"] = sp.csr_matrix(embeddings)

    max_copies = int(cfg["recommend"]["max_copies"])

    search_path = results / "weights_search.csv"
    search_emb_path = results / "weights_search_emb.csv"
    mmr_search_path = results / "mmr_lambda_search.csv"

    if args.skip_search:
        if not search_path.exists() or not search_emb_path.exists() or not mmr_search_path.exists():
            raise FileNotFoundError(
                "Arquivos de busca não encontrados. Execute sem --skip-search."
            )

        search = pd.read_csv(search_path)
        search_emb = pd.read_csv(search_emb_path)
        mmr_search = pd.read_csv(mmr_search_path)

    else:
        val = deck_data(
            split, "validation", len(catalog),
            max_copies, item_identity
        )

        # Block-weight search for TF-IDF, on validation decks only.
        search = search_weights(blocks, val, cfg, "busca de pesos TF-IDF")

        # Independent block-weight search for MiniLM.
        search_emb = search_weights(embedding_blocks, val, cfg, "busca de pesos MiniLM")

        weights = best_weights(search)
        print("Busca do λ do MMR na validação…")
        Xn_for_search = normalize_rows(compose(blocks, weights))
        mmr_search = search_mmr_lambda(
            Xn_for_search,
            val,
            [float(lam) for lam in cfg["evaluate"]["mmr_lambdas"]],
            int(cfg["evaluate"]["search_n"]),
            int(cfg["evaluate"]["batch_size"]),
        )

        search.to_csv(search_path, index=False)
        search_emb.to_csv(search_emb_path, index=False)
        mmr_search.to_csv(mmr_search_path, index=False)

        print(f"Busca TF-IDF: {len(search)} combinações")
        print(f"Busca MiniLM: {len(search_emb)} combinações")

    weights = best_weights(search)
    weights_emb = best_weights(search_emb)
    selected_lambda = best_mmr_lambda(mmr_search)

    n_s = cfg["evaluate"]["search_n"]

    print(
        f"Pesos TF-IDF: {weights} "
        f"(nDCG@{n_s} = {search.at[0, f'ndcg@{n_s}']:.4f})"
    )

    print(
        f"Pesos MiniLM: {weights_emb} "
        f"(nDCG@{n_s} = {search_emb.at[0, f'ndcg@{n_s}']:.4f})"
    )
    print(f"Lambda MMR escolhido na validação: {selected_lambda:.1f}")

    Xn = normalize_rows(compose(blocks, weights))

    Xn_emb = normalize_rows(compose(embedding_blocks, weights_emb))

    data = deck_data(
        split, "eval", len(catalog),
        max_copies, item_identity
    )

    n_values = [int(n) for n in cfg["recommend"]["n_values"]]

    # Top-N lists of every method on the eval decks.
    print(f"Avaliando {len(METHODS)} métodos em {len(data.deck_ids)} decks de avaliação…")
    recs = run_methods(
        Xn,
        data,
        split,
        item_identity,
        max(n_values),
        cfg,
        Xn_emb=Xn_emb,
        mmr_lambda=selected_lambda,
    )

    # Diversity is measured in one common space (the selected TF-IDF matrix) for every
    # method, so the column is comparable across methods, cbf_emb included.
    metrics = evaluate_methods(recs, data, Xn, n_values)

    metrics.to_csv(results / "metrics.csv", index=False)
    (results / "metrics.md").write_text(metrics_markdown(metrics, len(data.deck_ids)), encoding="utf-8")
    plot_metrics(metrics, max(n_values), results / "fig_metrics.png")
    params = run_params(
        cfg, weights, split, len(catalog), len(data.deck_ids),
        selected_lambda, weights_emb,
    )
    (results / "params.json").write_text(json.dumps(params, indent=2, ensure_ascii=False), encoding="utf-8")
    print(metrics.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    tradeoff_n = 10
    print("Curva de trade-off do MMR (nDCG@10 × diversidade) para cada λ…")
    eval_tradeoff_recs = {
        float(lam): rank_cbf_mmr(
            Xn, data, tradeoff_n, int(cfg["evaluate"]["batch_size"]), float(lam)
        )
        for lam in cfg["evaluate"]["mmr_lambdas"]
    }
    tradeoff = pd.DataFrame([
        {
            "lambda": lam,
            "ndcg@10": mean_ndcg(recs_at_lambda, data, tradeoff_n),
            "diversity": float(np.mean([
                intra_list_diversity([i for i, _ in rec], Xn)
                for rec in recs_at_lambda
            ])),
        }
        for lam, recs_at_lambda in eval_tradeoff_recs.items()
    ])
    tradeoff.to_csv(results / "mmr_tradeoff.csv", index=False)
    plot_mmr_tradeoff(
        tradeoff, selected_lambda, results / "fig_mmr_tradeoff.png"
    )

    ev = cfg["evaluate"]
    per_deck = per_deck_metrics(recs[ev["qualitative_method"]], data, int(ev["qualitative_n"]))
    report = qualitative_report(split, catalog, Xn, feature_names(block_names), per_deck, item_identity, cfg)
    (results / "qualitative.md").write_text(report, encoding="utf-8")
    print(f"Relatório qualitativo: {results / 'qualitative.md'}")


if __name__ == "__main__":
    main()
