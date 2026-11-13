"""Streamlit app — recomendação de cartas de Magic: The Gathering.

Interface interativa com duas abas:
1. **Deck do dataset**: seleciona um deck de avaliação e exibe recomendações.
2. **Monte seu deck**: o usuário monta um deck customizado e recebe recomendações.

Execução::

    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy.sparse as sp
import streamlit as st

# ── Imports do projeto ──────────────────────────────────────────────────────
from src.config import load_config
from src.explain import Explanation, explain, item_attributes
from src.features import BLOCKS, compose, feature_names, load_blocks
from src.profile import build_profile, copy_weights
from src.recommend import (
    deck_identity,
    identity_bits,
    normalize_rows,
    recommend,
)
from src.split import load_split

# ── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(page_title="🧙 MTG Card Recommender", layout="wide")

N_COLS = 5  # colunas do grid de cartas


# ── Carregamento de dados (cacheado) ────────────────────────────────────────
@st.cache_resource(show_spinner="Carregando dados do catálogo…")
def load_data() -> dict[str, Any]:
    """Carrega todos os artefatos pré-computados necessários pela app."""
    cfg = load_config()
    processed = Path(cfg["paths"]["processed"])

    # Catálogo e split
    catalog = pd.read_parquet(processed / "catalog.parquet")
    split = load_split(processed)

    # Blocos TF-IDF pré-computados
    blocks_mat, blocks_names = load_blocks(processed)

    # Matriz composta TF-IDF (row-normalized)
    item_matrix = sp.load_npz(processed / "item_matrix.npz").tocsr()
    Xn = normalize_rows(item_matrix)

    # Nomes das features alinhados com a matriz composta
    fnames = feature_names(blocks_names)

    # Embeddings (opcional — pode não existir)
    emb_path = processed / "embeddings.npy"
    Xn_emb: sp.csr_matrix | None = None
    if emb_path.exists():
        embeddings = np.load(emb_path).astype(np.float64)
        emb_sparse = sp.csr_matrix(embeddings)
        # Substituir bloco 'text' pelos embeddings e recompor
        emb_blocks = dict(blocks_mat)
        emb_blocks["text"] = emb_sparse
        weights = cfg["features"]["block_weights"]
        emb_matrix = compose(emb_blocks, weights)
        Xn_emb = normalize_rows(emb_matrix)
        embedding_names = [f"embedding:{i}" for i in range(embeddings.shape[1])]
        fnames_emb = embedding_names + [
            name for block in BLOCKS[1:] for name in blocks_names[block]
        ]
    else:
        fnames_emb = fnames

    # Identidade de cor de cada carta
    item_identity = identity_bits(catalog.sort_values("item_idx")["color_identity"])

    # Atributos para as explicações
    item_keywords, item_types = item_attributes(catalog)

    return {
        "cfg": cfg,
        "catalog": catalog.sort_values("item_idx").reset_index(drop=True),
        "split": split,
        "Xn": Xn,
        "Xn_emb": Xn_emb,
        "fnames": fnames,
        "fnames_emb": fnames_emb,
        "item_identity": item_identity,
        "item_keywords": item_keywords,
        "item_types": item_types,
    }


# ── Helpers ─────────────────────────────────────────────────────────────────
def _get_matrix(data: dict[str, Any], use_emb: bool) -> sp.csr_matrix:
    """Retorna a matriz normalizada de acordo com a representação escolhida."""
    if use_emb and data["Xn_emb"] is not None:
        return data["Xn_emb"]
    return data["Xn"]


def _render_card(
    catalog: pd.DataFrame,
    idx: int,
    score: float,
    exp: Explanation | None,
) -> None:
    """Renderiza uma carta: imagem, nome, score e expansor de explicação."""
    row = catalog.iloc[idx]
    name = row["name"]
    img_url = row.get("image_url", "")

    if img_url:
        st.image(img_url, width=200)
    st.markdown(f"**{name}**")
    st.caption(f"Score: {score:.4f}")

    if exp is not None:
        with st.expander("🔍 Explicação"):
            # Top features
            if exp.top_features:
                st.markdown("**Features mais relevantes:**")
                for fc in exp.top_features[:3]:
                    st.markdown(f"- {fc.label} ({fc.pct:.0f}%)")

            # Carta mais parecida
            similar_name = catalog.iloc[exp.similar_item_idx]["name"]
            st.markdown(
                f"**Carta do deck mais parecida:** {similar_name} "
                f"(cosseno {exp.similar_cosine:.2f})"
            )

            # Keywords e types compartilhados
            if exp.shared_keywords:
                kw_str = ", ".join(f"{k} ({n})" for k, n in exp.shared_keywords[:5])
                st.markdown(f"**Palavras-chave em comum:** {kw_str}")
            if exp.shared_types:
                tp_str = ", ".join(f"{t} ({n})" for t, n in exp.shared_types[:5])
                st.markdown(f"**Tipos em comum:** {tp_str}")


def _render_recommendations(
    recs: list[tuple[int, float]],
    profile: np.ndarray,
    train_idxs: np.ndarray,
    data: dict[str, Any],
    matrix: sp.csr_matrix,
) -> None:
    """Exibe as recomendações em um grid de N_COLS colunas."""
    catalog = data["catalog"]
    fnames = data["fnames_emb"] if matrix is data["Xn_emb"] else data["fnames"]
    item_keywords = data["item_keywords"]
    item_types = data["item_types"]
    cfg = data["cfg"]
    k_explain = int(cfg["recommend"]["explain_top_features"])

    if not recs:
        st.info("Nenhuma recomendação gerada.")
        return

    for row_start in range(0, len(recs), N_COLS):
        cols = st.columns(N_COLS)
        for col, (idx, score) in zip(cols, recs[row_start : row_start + N_COLS]):
            with col:
                # Explicação — usa a mesma matriz da recomendação
                exp = explain(
                    item_idx=idx,
                    profile=profile,
                    item_matrix=matrix,
                    train_idxs=train_idxs,
                    names=fnames,
                    item_keywords=item_keywords,
                    item_types=item_types,
                    k=k_explain,
                )
                _render_card(catalog, idx, score, exp)


# ── Sidebar ─────────────────────────────────────────────────────────────────
st.sidebar.title("⚙️ Configurações")

n_recs = st.sidebar.slider("Número de recomendações (N)", min_value=1, max_value=30, value=10)
color_filter = st.sidebar.checkbox("Filtrar por identidade de cor do deck", value=False)
representation = st.sidebar.radio(
    "Representação",
    options=["TF-IDF", "Embeddings (MiniLM)"],
    index=0,
)
mmr_enabled = st.sidebar.checkbox("MMR (diversidade)", value=False)
mmr_lambda: float | None = None
if mmr_enabled:
    mmr_lambda = st.sidebar.slider("λ (lambda)", min_value=0.0, max_value=1.0, value=0.7, step=0.05)

# ── Carregar dados ──────────────────────────────────────────────────────────
data = load_data()
catalog = data["catalog"]
split = data["split"]
cfg = data["cfg"]
use_emb = representation == "Embeddings (MiniLM)"
matrix = _get_matrix(data, use_emb)

# Avisar se embeddings não estão disponíveis
if use_emb and data["Xn_emb"] is None:
    st.sidebar.warning("⚠️ Embeddings não encontrados. Usando TF-IDF.")
    matrix = data["Xn"]

# ── Título ──────────────────────────────────────────────────────────────────
st.title("🧙 MTG Card Recommender")

# ── Abas ────────────────────────────────────────────────────────────────────
tab_dataset, tab_custom = st.tabs(["Deck do dataset", "Monte seu deck"])

# ---- Aba 1: Deck do dataset ------------------------------------------------
with tab_dataset:
    # Decks de avaliação disponíveis
    eval_decks = split[split["subset"] == "eval"]["deck_id"].unique()
    if len(eval_decks) == 0:
        st.warning("Nenhum deck de avaliação encontrado no split.")
    else:
        selected_deck = st.selectbox(
            "Selecione um deck de avaliação:",
            sorted(eval_decks),
            index=0,
        )

        deck = split[split["deck_id"] == selected_deck]
        train = deck[deck["part"] == "train"]
        train_idxs = train["item_idx"].to_numpy(dtype=np.int64)

        # Mostrar cartas de treino
        with st.expander("📋 Cartas do deck (treino)", expanded=False):
            train_names = [
                catalog.iloc[i]["name"] for i in train_idxs
            ]
            for name in sorted(train_names):
                st.markdown(f"- {name}")

        # Construir perfil e recomendar
        max_copies = int(cfg["recommend"]["max_copies"])
        weights = copy_weights(train["copies"].to_numpy(), max_copies)
        profile = build_profile(matrix, train_idxs, weights)

        # Identidade de cor (se filtro ativo)
        color_id = None
        item_identity = data["item_identity"]
        if color_filter:
            color_id = deck_identity(item_identity, train_idxs)

        recs = recommend(
            profile=profile,
            item_matrix=matrix,
            exclude_idxs=train_idxs,
            n=n_recs,
            color_identity=color_id,
            mmr=mmr_lambda,
            item_identity=item_identity if color_filter else None,
        )

        st.subheader(f"Top-{n_recs} recomendações")
        _render_recommendations(recs, profile, train_idxs, data, matrix)

# ---- Aba 2: Monte seu deck -------------------------------------------------
with tab_custom:
    all_names = sorted(catalog["name"].tolist())
    selected_names = st.multiselect(
        "Escolha cartas para o seu deck:",
        options=all_names,
        default=[],
        placeholder="Pesquise e selecione cartas…",
    )

    if not selected_names:
        st.info("Selecione ao menos uma carta para receber recomendações.")
    else:
        # Mapear nomes → item_idx
        name_to_idx = dict(zip(catalog["name"], catalog["item_idx"]))
        custom_idxs = np.array(
            [name_to_idx[n] for n in selected_names if n in name_to_idx],
            dtype=np.int64,
        )

        if custom_idxs.size == 0:
            st.warning("Nenhuma das cartas selecionadas foi encontrada no catálogo.")
        else:
            # Todas as cópias = 1
            custom_weights = np.ones(len(custom_idxs), dtype=np.float64)
            profile = build_profile(matrix, custom_idxs, custom_weights)

            # Identidade de cor (se filtro ativo)
            color_id = None
            item_identity = data["item_identity"]
            if color_filter:
                color_id = deck_identity(item_identity, custom_idxs)

            recs = recommend(
                profile=profile,
                item_matrix=matrix,
                exclude_idxs=custom_idxs,
                n=n_recs,
                color_identity=color_id,
                mmr=mmr_lambda,
                item_identity=item_identity if color_filter else None,
            )

            st.subheader(f"Top-{n_recs} recomendações")
            _render_recommendations(recs, profile, custom_idxs, data, matrix)

# ── Rodapé ──────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "Magic: The Gathering é propriedade da Wizards of the Coast. "
    "Imagens de cartas: [Scryfall](https://scryfall.com)."
)
