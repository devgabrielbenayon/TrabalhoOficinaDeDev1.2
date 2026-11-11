"""Explanations computed from the very vectors used by the ranking.

The score of a recommendation is ``cos(profile, item) = sum_k profile[k] * item[k]``
(both unit vectors). Each term of that sum is the contribution of feature ``k``; the
explanation lists the largest terms, so it is a faithful decomposition of the score and
not a post-hoc story. Two extra reasons are derived from the same matrix and metadata:
the most similar train card and the keywords/types shared with the deck.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import pandas as pd
import scipy.sparse as sp

from src.features import type_tokens

PREFIX_LABELS = {
    "text": "texto", "type": "tipo", "kw": "palavra-chave",
    "color": "cor", "identity": "identidade de cor", "num": "atributo numérico",
}


@dataclass
class FeatureContribution:
    """One term of the dot product."""

    name: str           # raw column name, e.g. "text:any target"
    label: str          # human-readable name
    contribution: float  # profile[k] * item[k]
    pct: float          # share of the score, in %


@dataclass
class Explanation:
    """The three reasons behind one recommendation."""

    item_idx: int
    score: float
    top_features: list[FeatureContribution]
    similar_item_idx: int
    similar_cosine: float
    shared_keywords: list[tuple[str, int]] = field(default_factory=list)  # (keyword, n train cards)
    shared_types: list[tuple[str, int]] = field(default_factory=list)     # (type, n train cards)


def readable(name: str) -> str:
    """'text:any target' -> 'texto: "any target"'; 'kw:flying' -> 'palavra-chave: flying'."""
    prefix, _, value = name.partition(":")
    label = PREFIX_LABELS.get(prefix, prefix)
    return f'{label}: "{value}"' if prefix == "text" else f"{label}: {value}"


def contributions(profile: np.ndarray, item_row: sp.csr_matrix) -> np.ndarray:
    """Dense vector of ``profile[k] * item[k]``; it sums to the cosine score."""
    return np.asarray(item_row.toarray()).ravel() * np.asarray(profile, dtype=np.float64)


def item_attributes(catalog: pd.DataFrame) -> tuple[list[set[str]], list[set[str]]]:
    """Lower-cased keyword and type sets of each catalog row (aligned with ``item_idx``)."""
    catalog = catalog.sort_values("item_idx")
    keywords = [{k.lower() for k in ks} for ks in catalog["keywords"]]
    types = [type_tokens(t) for t in catalog["type_line"]]
    return keywords, types


def _shared(item_attrs: set[str], train_attrs: Sequence[set[str]]) -> list[tuple[str, int]]:
    """Attributes of the item also found in train cards, with how many train cards have them."""
    counts = Counter(a for attrs in train_attrs for a in attrs & item_attrs)
    return sorted(counts.items(), key=lambda x: (-x[1], x[0]))


def explain(
    item_idx: int,
    profile: np.ndarray,
    item_matrix: sp.csr_matrix,
    train_idxs: Sequence[int] | np.ndarray,
    names: Sequence[str],
    item_keywords: Sequence[set[str]],
    item_types: Sequence[set[str]],
    k: int,
) -> Explanation:
    """Explain why ``item_idx`` was recommended for the deck with ``profile``.

    ``item_matrix`` must be the same row-normalized matrix used by ``recommend``.
    """
    row = item_matrix[item_idx]
    contrib = contributions(profile, row)
    score = float(contrib.sum())
    order = np.argsort(-contrib, kind="stable")[:k]
    top = [
        FeatureContribution(names[j], readable(names[j]), float(contrib[j]),
                            100.0 * float(contrib[j]) / score if score else 0.0)
        for j in order if contrib[j] > 0
    ]

    train = np.asarray(list(train_idxs), dtype=np.int64)
    sims = np.asarray(item_matrix[train] @ row.T.toarray()).ravel()
    best = int(np.argmax(sims))

    return Explanation(
        item_idx=int(item_idx),
        score=score,
        top_features=top,
        similar_item_idx=int(train[best]),
        similar_cosine=float(sims[best]),
        shared_keywords=_shared(item_keywords[item_idx], [item_keywords[t] for t in train]),
        shared_types=_shared(item_types[item_idx], [item_types[t] for t in train]),
    )


def format_explanation(exp: Explanation, card_names: Sequence[str]) -> str:
    """Multi-line, Portuguese text of an explanation (used by the demo and the reports)."""
    feats = "; ".join(f"{f.label} ({f.pct:.0f}%)" for f in exp.top_features)
    kws = ", ".join(f"{a} ({n})" for a, n in exp.shared_keywords) or "nenhuma"
    tps = ", ".join(f"{a} ({n})" for a, n in exp.shared_types) or "nenhum"
    return (
        f"    features que mais contribuíram: {feats}\n"
        f"    carta do deck mais parecida: {card_names[exp.similar_item_idx]} "
        f"(cosseno {exp.similar_cosine:.2f})\n"
        f"    em comum com o deck: palavras-chave {kws}; tipos {tps}"
    )
