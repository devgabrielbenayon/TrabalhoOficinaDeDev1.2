"""Streamlit app: Magic: The Gathering card recommendations with explanations.

Two tabs:
1. **Deck do dataset**: pick an evaluation deck and see its recommendations.
2. **Monte seu deck**: pick cards from the catalog and get recommendations on the fly.

The item matrices use the block weights selected on the validation decks
(``results/params.json``, written by ``python -m src.evaluate``), so the app ranks
exactly like the evaluated methods. Without that file it falls back to ``config.yaml``.

Run from the project root::

    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

# `streamlit run` only puts app/ on sys.path; add the project root so `src` imports work.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import scipy.sparse as sp  # noqa: E402
import streamlit as st  # noqa: E402

from src.config import load_config  # noqa: E402
from src.explain import Explanation, explain, item_attributes  # noqa: E402
from src.features import BLOCKS, compose, feature_names, load_blocks  # noqa: E402
from src.profile import build_profile, copy_weights  # noqa: E402
from src.recommend import (  # noqa: E402
    COLOR_BITS,
    deck_identity,
    identity_bits,
    normalize_rows,
    recommend,
)
from src.split import load_split  # noqa: E402

st.set_page_config(page_title="MTG Card Recommender", page_icon="🧙", layout="wide")

N_COLS = 5  # cards per row in the recommendation grid


# ── Data loading (cached) ───────────────────────────────────────────────────
def selected_weights(cfg: dict[str, Any]) -> tuple[dict[str, float], dict[str, float], bool]:
    """Block weights chosen on validation (TF-IDF, MiniLM); falls back to config.yaml."""
    params_path = Path(cfg["paths"]["results"]) / "params.json"
    if params_path.exists():
        params = json.loads(params_path.read_text(encoding="utf-8"))
        tfidf = params.get("block_weights_selected")
        emb = params.get("embedding_block_weights_selected")
        if tfidf and emb:
            return tfidf, emb, True
    default = cfg["features"]["block_weights"]
    return default, default, False


def color_label(bits: int) -> str:
    """Color identity bits -> 'WUBRG'-style string ('incolor' when empty)."""
    return "".join(c for c, b in COLOR_BITS.items() if bits & b) or "incolor"


@st.cache_resource(show_spinner="Carregando dados do catálogo…")
def load_data() -> dict[str, Any]:
    """Load every precomputed artifact the app needs (no feature recomputation)."""
    cfg = load_config()
    processed = Path(cfg["paths"]["processed"])

    catalog = pd.read_parquet(processed / "catalog.parquet").sort_values("item_idx")
    catalog = catalog.reset_index(drop=True)
    split = load_split(processed)
    blocks, block_names = load_blocks(processed)
    weights, weights_emb, from_validation = selected_weights(cfg)

    Xn = normalize_rows(compose(blocks, weights))
    fnames = feature_names(block_names)

    # MiniLM embeddings replace only the text block; the other blocks stay.
    Xn_emb, fnames_emb = None, fnames
    emb_path = processed / "embeddings.npy"
    if emb_path.exists():
        embeddings = np.load(emb_path).astype(np.float64)
        emb_blocks = dict(blocks) | {"text": sp.csr_matrix(embeddings)}
        Xn_emb = normalize_rows(compose(emb_blocks, weights_emb))
        fnames_emb = [f"emb:{i}" for i in range(embeddings.shape[1])] + [
            name for block in BLOCKS[1:] for name in block_names[block]
        ]

    item_identity = identity_bits(catalog["color_identity"])
    item_keywords, item_types = item_attributes(catalog)

    # Human-readable label of each evaluation deck: colors, event and date.
    eval_split = split[split["subset"] == "eval"]
    meta = pd.read_parquet(processed / "deck_meta.parquet").set_index("deck_id")
    deck_labels = {}
    for deck_id, rows in eval_split[eval_split["part"] == "train"].groupby("deck_id"):
        colors = color_label(deck_identity(item_identity, rows["item_idx"].to_numpy()))
        event = f" · {meta.at[deck_id, 'event']} · {meta.at[deck_id, 'date']}" if deck_id in meta.index else ""
        deck_labels[int(deck_id)] = f"Deck {deck_id} · {colors}{event}"

    return {
        "cfg": cfg,
        "catalog": catalog,
        "split": split,
        "Xn": Xn,
        "Xn_emb": Xn_emb,
        "fnames": fnames,
        "fnames_emb": fnames_emb,
        "item_identity": item_identity,
        "item_keywords": item_keywords,
        "item_types": item_types,
        "deck_labels": deck_labels,
        "weights_from_validation": from_validation,
    }


# ── Rendering helpers ───────────────────────────────────────────────────────
def render_card(catalog: pd.DataFrame, idx: int, score: float, exp: Explanation) -> None:
    """One card: image, name, score and an expander with the three explanations."""
    row = catalog.iloc[idx]
    if row.get("image_url"):
        st.image(row["image_url"], width=200)
    st.markdown(f"**{row['name']}**")
    st.caption(f"Score (cosseno): {score:.4f}")

    with st.expander("🔍 Por que esta carta?"):
        if exp.top_features:
            st.markdown("**Features que mais contribuíram para o score:**")
            for fc in exp.top_features:
                st.markdown(f"- {fc.label} ({fc.pct:.0f}%)")
        similar = catalog.iloc[exp.similar_item_idx]["name"]
        st.markdown(f"**Carta do deck mais parecida:** {similar} (cosseno {exp.similar_cosine:.2f})")
        if exp.shared_keywords:
            kws = ", ".join(f"{k} ({n})" for k, n in exp.shared_keywords[:5])
            st.markdown(f"**Palavras-chave em comum:** {kws}")
        if exp.shared_types:
            tps = ", ".join(f"{t} ({n})" for t, n in exp.shared_types[:5])
            st.markdown(f"**Tipos em comum:** {tps}")


def render_recommendations(
    recs: list[tuple[int, float]],
    profile: np.ndarray,
    train_idxs: np.ndarray,
    data: dict[str, Any],
    matrix: sp.csr_matrix,
    use_emb: bool,
) -> None:
    """Grid of recommendations; explanations use the same matrix as the ranking."""
    if not recs:
        st.info("Nenhuma recomendação gerada.")
        return
    fnames = data["fnames_emb"] if use_emb else data["fnames"]
    k = int(data["cfg"]["recommend"]["explain_top_features"])
    for start in range(0, len(recs), N_COLS):
        for col, (idx, score) in zip(st.columns(N_COLS), recs[start:start + N_COLS]):
            with col:
                exp = explain(idx, profile, matrix, train_idxs, fnames,
                              data["item_keywords"], data["item_types"], k)
                render_card(data["catalog"], idx, score, exp)


def show_top_n(train_idxs: np.ndarray, weights: np.ndarray, data: dict[str, Any], opts: dict[str, Any]) -> None:
    """Build the profile from the given cards, rank the catalog and render the Top-N."""
    matrix = opts["matrix"]
    profile = build_profile(matrix, train_idxs, weights)
    color_id = deck_identity(data["item_identity"], train_idxs) if opts["color_filter"] else None
    recs = recommend(
        profile, matrix, train_idxs, opts["n"],
        color_identity=color_id, mmr=opts["mmr"],
        item_identity=data["item_identity"] if opts["color_filter"] else None,
    )
    st.subheader(f"Top-{opts['n']} recomendações")
    render_recommendations(recs, profile, train_idxs, data, matrix, opts["use_emb"])


# ── Sidebar ─────────────────────────────────────────────────────────────────
st.sidebar.title("⚙️ Configurações")
n_recs = st.sidebar.slider("Número de recomendações (N)", min_value=1, max_value=30, value=10)
color_filter = st.sidebar.checkbox("Filtrar por identidade de cor do deck", value=False)
representation = st.sidebar.radio("Representação do texto", ["TF-IDF", "Embeddings (MiniLM)"], index=0)
mmr_enabled = st.sidebar.checkbox("MMR (diversidade)", value=False)
mmr_lambda = (
    st.sidebar.slider("λ (1 = só relevância)", min_value=0.0, max_value=1.0, value=0.7, step=0.05)
    if mmr_enabled else None
)

data = load_data()
catalog, split, cfg = data["catalog"], data["split"], data["cfg"]
use_emb = representation == "Embeddings (MiniLM)" and data["Xn_emb"] is not None
if representation == "Embeddings (MiniLM)" and data["Xn_emb"] is None:
    st.sidebar.warning("⚠️ Embeddings não encontrados (rode `python run_all.py`). Usando TF-IDF.")
if not data["weights_from_validation"]:
    st.sidebar.warning("⚠️ `results/params.json` não encontrado: usando os pesos padrão do config.yaml.")

options = {
    "matrix": data["Xn_emb"] if use_emb else data["Xn"],
    "use_emb": use_emb,
    "n": n_recs,
    "color_filter": color_filter,
    "mmr": mmr_lambda,
}

st.title("🧙 MTG Card Recommender")
tab_dataset, tab_custom = st.tabs(["Deck do dataset", "Monte seu deck"])

# ---- Tab 1: evaluation deck --------------------------------------------------
with tab_dataset:
    deck_labels = data["deck_labels"]
    if not deck_labels:
        st.warning("Nenhum deck de avaliação encontrado no split.")
    else:
        selected_deck = st.selectbox(
            "Selecione um deck de avaliação:", sorted(deck_labels), index=0,
            format_func=lambda d: deck_labels[d],
        )
        deck = split[split["deck_id"] == selected_deck]
        train = deck[deck["part"] == "train"].sort_values(["copies", "card_name"], ascending=[False, True])
        with st.expander(f"📋 Cartas do deck usadas no perfil ({len(train)})", expanded=False):
            for r in train.itertuples():
                st.markdown(f"- {r.copies}x {r.card_name}")
        max_copies = int(cfg["recommend"]["max_copies"])
        show_top_n(train["item_idx"].to_numpy(dtype=np.int64),
                   copy_weights(train["copies"].to_numpy(), max_copies), data, options)

# ---- Tab 2: custom deck ------------------------------------------------------
with tab_custom:
    selected_names = st.multiselect(
        "Escolha cartas para o seu deck:", options=sorted(catalog["name"]),
        default=[], placeholder="Pesquise e selecione cartas…",
    )
    if not selected_names:
        st.info("Selecione ao menos uma carta para receber recomendações.")
    else:
        name_to_idx = dict(zip(catalog["name"], catalog["item_idx"]))
        custom_idxs = np.array([name_to_idx[n] for n in selected_names], dtype=np.int64)
        # Every chosen card counts the same (as a full playset).
        show_top_n(custom_idxs, np.ones(len(custom_idxs)), data, options)

# ---- Footer ------------------------------------------------------------------
st.markdown("---")
st.caption(
    "Conteúdo de fã, sem afiliação com a Wizards of the Coast. Magic: The Gathering é "
    "propriedade da Wizards of the Coast. Imagens e dados de cartas: [Scryfall](https://scryfall.com)."
)
