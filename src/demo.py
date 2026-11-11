"""Print the Top-N recommendations with explanations for one deck of the split.

Usage: python -m src.demo [--deck DECK_ID] [--color]
Requires ``python -m src.features`` and ``python -m src.split`` to have been run.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import scipy.sparse as sp

from src.config import load_config
from src.explain import explain, format_explanation, item_attributes
from src.features import feature_names, load_blocks
from src.recommend import identity_bits, normalize_rows, recommend_for_deck
from src.split import load_split


def render_deck(
    deck: pd.DataFrame,
    X: sp.csr_matrix,
    catalog: pd.DataFrame,
    names: list[str],
    cfg: dict,
    n: int,
    color_filter: bool,
) -> str:
    """Text block with the train/test cards and the explained Top-N of one deck."""
    rc = cfg["recommend"]
    card_names = catalog.sort_values("item_idx")["name"].tolist()
    identity = identity_bits(catalog.sort_values("item_idx")["color_identity"])
    keywords, types = item_attributes(catalog)

    recs, profile, train_idxs = recommend_for_deck(
        deck, X, n, int(rc["max_copies"]), identity, color_filter
    )
    test = set(deck.loc[deck["part"] == "test", "item_idx"])
    lines = [
        f"Deck {int(deck['deck_id'].iloc[0])} ({deck['subset'].iloc[0]})",
        f"  treino ({len(train_idxs)}): " + ", ".join(sorted(card_names[i] for i in train_idxs)),
        f"  teste ({len(test)}): " + ", ".join(sorted(card_names[i] for i in test)),
        f"  Top-{n}{' (filtro de cor)' if color_filter else ''}, acertos: "
        f"{sum(i in test for i, _ in recs)}",
    ]
    for rank, (i, score) in enumerate(recs, 1):
        mark = "✔" if i in test else " "
        lines.append(f"{rank:>3}. {mark} {card_names[i]}  (score {score:.3f})")
        exp = explain(i, profile, X, train_idxs, names, keywords, types, int(rc["explain_top_features"]))
        lines.append(format_explanation(exp, card_names))
    return "\n".join(lines)


def main() -> None:
    """Load the artifacts and print the demo deck."""
    cfg = load_config()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deck", type=int, default=cfg["recommend"]["demo_deck_id"])
    parser.add_argument("--color", action="store_true", default=cfg["recommend"]["color_filter"])
    args = parser.parse_args()

    processed = Path(cfg["paths"]["processed"])
    catalog = pd.read_parquet(processed / "catalog.parquet")
    split = load_split(processed)
    _, block_names = load_blocks(processed)
    names = feature_names(block_names)
    X = normalize_rows(sp.load_npz(processed / "item_matrix.npz"))

    deck_id = args.deck
    if deck_id is None:
        deck_id = int(split.loc[split["subset"] == "eval", "deck_id"].min())
    deck = split[split["deck_id"] == deck_id]
    if deck.empty:
        raise SystemExit(f"Deck {deck_id} não está no split.")
    print(render_deck(deck, X, catalog, names, cfg, int(cfg["recommend"]["demo_n"]), args.color))


if __name__ == "__main__":
    main()
