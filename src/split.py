"""Per-deck train/test split with a fixed seed.

Protocol:
1. Decks with fewer than ``split.min_cards_per_deck`` distinct (non-basic) cards are
   dropped from the evaluation; how many is recorded in ``split_report.json``.
2. A fraction ``split.validation_fraction`` of the remaining decks forms the
   ``validation`` subset (used only to tune block weights / MMR lambda); the rest is ``eval``.
3. Inside every deck, ``split.test_fraction`` of the distinct cards (at least 1) is held
   out as ``test``; the rest is ``train`` (the only part a profile may be built from).

Output: ``data/processed/split.parquet`` with columns
``deck_id, card_name, item_idx, copies, part, subset``.

Usage: python -m src.split
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config import load_config, make_rng

SPLIT_COLUMNS = ["deck_id", "card_name", "item_idx", "copies", "part", "subset"]


def n_test_cards(n_distinct: int, test_fraction: float) -> int:
    """Number of held-out cards: ``round(fraction * n)``, at least 1 and leaving 1 for train."""
    return int(min(max(1, round(test_fraction * n_distinct)), n_distinct - 1))


def split_decks(
    decks: pd.DataFrame, catalog: pd.DataFrame, cfg: dict[str, Any]
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Split ``decks`` (``deck_id, card_name, copies``) into train/test and validation/eval.

    Deterministic for a given ``cfg["seed"]``: decks and cards are sorted before sampling.
    """
    p = cfg["split"]
    rng = make_rng(cfg, cfg["seed_offsets"]["split"])
    name_to_idx = dict(zip(catalog["name"], catalog["item_idx"]))

    df = decks[["deck_id", "card_name", "copies"]].copy()
    df["item_idx"] = df["card_name"].map(name_to_idx)
    df = df.dropna(subset=["item_idx"]).astype({"item_idx": int})
    df = df.groupby(["deck_id", "card_name", "item_idx"], as_index=False)["copies"].sum()
    df = df.sort_values(["deck_id", "card_name"]).reset_index(drop=True)

    sizes = df.groupby("deck_id").size()
    kept_ids = np.sort(sizes.index[sizes >= int(p["min_cards_per_deck"])].to_numpy())
    df = df[df["deck_id"].isin(kept_ids)].reset_index(drop=True)

    n_val = int(round(float(p["validation_fraction"]) * len(kept_ids)))
    val_ids = set(rng.permutation(kept_ids)[:n_val].tolist())
    df["subset"] = np.where(df["deck_id"].isin(val_ids), "validation", "eval")

    part = np.empty(len(df), dtype=object)
    for _, rows in df.groupby("deck_id", sort=True).indices.items():
        k = n_test_cards(len(rows), float(p["test_fraction"]))
        test_rows = rng.permutation(rows)[:k]
        part[rows] = "train"
        part[test_rows] = "test"
    df["part"] = part

    report = {
        "decks_input": int(decks["deck_id"].nunique()),
        "decks_dropped_small": int((sizes < int(p["min_cards_per_deck"])).sum()),
        "decks_kept": int(len(kept_ids)),
        "decks_validation": len(val_ids),
        "decks_eval": int(len(kept_ids) - len(val_ids)),
        "test_fraction_observed": float((df["part"] == "test").mean()) if len(df) else 0.0,
    }
    return df[SPLIT_COLUMNS], report


def load_split(processed: str | Path) -> pd.DataFrame:
    """Read ``split.parquet``."""
    return pd.read_parquet(Path(processed) / "split.parquet")


def main() -> None:
    """Build and save the split, print a short report."""
    cfg = load_config()
    processed = Path(cfg["paths"]["processed"])
    decks = pd.read_parquet(processed / "decks.parquet")
    catalog = pd.read_parquet(processed / "catalog.parquet")

    split, report = split_decks(decks, catalog, cfg)
    split.to_parquet(processed / "split.parquet", index=False)
    (processed / "split_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
