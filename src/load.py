"""Load Scryfall cards and MTGO decklists into tidy tables and build the item catalog.

Outputs (in ``paths.processed``): ``cards.parquet``, ``decks.parquet``,
``deck_meta.parquet``, ``catalog.parquet`` and ``load_report.json``; plus
``results/unmatched_cards.csv``.

Usage: python -m src.load
"""

from __future__ import annotations

import gzip
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterator

import pandas as pd
import requests

from src.config import load_config

# Layouts that are not real playable cards (art cards, tokens, emblems, ...).
NON_CARD_LAYOUTS = {
    "art_series", "token", "double_faced_token", "emblem",
    "vanguard", "scheme", "planar", "augment", "host",
}
CARD_COLUMNS = [
    "name", "front_name", "oracle_text", "type_line", "mana_cost", "cmc", "colors",
    "color_identity", "keywords", "power", "toughness", "layout", "legality", "image_url",
]


# --------------------------------------------------------------------------- cards
def is_basic_land(type_line: str) -> bool:
    """True for basic lands, including Snow-Covered basics ("Basic Snow Land") and Wastes."""
    front = (type_line or "").split(" // ")[0]
    supertypes_and_types = front.split("—")[0].split()
    return "Basic" in supertypes_and_types and "Land" in supertypes_and_types


def _face_value(card: dict[str, Any], key: str) -> Any:
    """Top-level value or, for multi-face cards, the first face's value."""
    if card.get(key) is not None:
        return card[key]
    for face in card.get("card_faces", []):
        if face.get(key) is not None:
            return face[key]
    return None


def _image_url(card: dict[str, Any]) -> str | None:
    """'normal' image URL of the card or of its first face that has an image."""
    if card.get("image_uris"):
        return card["image_uris"].get("normal")
    for face in card.get("card_faces", []):
        if face.get("image_uris"):
            return face["image_uris"].get("normal")
    return None


def _parse_card(card: dict[str, Any], fmt: str) -> dict[str, Any]:
    """Flatten one Scryfall oracle card into the columns we use."""
    faces = card.get("card_faces") or []
    if faces:
        oracle_text = "\n//\n".join(f.get("oracle_text", "") for f in faces)
        type_line = card.get("type_line") or " // ".join(f.get("type_line", "") for f in faces)
        colors = card.get("colors")
        if colors is None:
            colors = sorted({c for f in faces for c in f.get("colors", [])})
    else:
        oracle_text = card.get("oracle_text", "")
        type_line = card.get("type_line", "")
        colors = card.get("colors", [])
    return {
        "name": card["name"],
        "front_name": card["name"].split(" // ")[0],
        "oracle_text": oracle_text or "",
        "type_line": type_line or "",
        "mana_cost": _face_value(card, "mana_cost") or "",
        "cmc": float(card.get("cmc") or 0.0),
        "colors": list(colors or []),
        "color_identity": list(card.get("color_identity", [])),
        "keywords": list(card.get("keywords", [])),
        "power": _face_value(card, "power"),
        "toughness": _face_value(card, "toughness"),
        "layout": card.get("layout", ""),
        "legality": card.get("legalities", {}).get(fmt.lower(), "not_legal"),
        "image_url": _image_url(card),
    }


def load_cards(path: str | Path, fmt: str = "Modern") -> pd.DataFrame:
    """Read the Scryfall oracle bulk (JSONL.gz or JSON) into one row per playable card name."""
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    rows = []
    with opener(path, "rt", encoding="utf-8") as fh:
        if ".jsonl" in path.name:
            raw = (json.loads(line) for line in fh if line.strip())
        else:
            raw = iter(json.load(fh))
        for card in raw:
            if card.get("layout") in NON_CARD_LAYOUTS:
                continue
            rows.append(_parse_card(card, fmt))
    df = pd.DataFrame(rows, columns=CARD_COLUMNS)
    # Rare name collisions (e.g. un-set reprints): keep the version legal in the format.
    df["_legal_rank"] = (df["legality"] != "legal").astype(int)
    df = df.sort_values(["name", "_legal_rank"]).drop_duplicates("name").drop(columns="_legal_rank")
    return df.reset_index(drop=True)


# --------------------------------------------------------------------------- names
def normalize_deck_name(name: str) -> str:
    """Normalize a decklist name: trim and turn 'A/B' split notation into 'A // B'."""
    name = " ".join(name.strip().split())
    if "/" in name and " // " not in name:
        name = " // ".join(part.strip() for part in name.split("/"))
    return name


def build_name_index(cards: pd.DataFrame) -> dict[str, str]:
    """Map lower-cased full names and front-face names to the canonical Scryfall name."""
    index = {name.lower(): name for name in cards["name"]}
    for full, front in zip(cards["name"], cards["front_name"]):
        index.setdefault(front.lower(), full)  # never override an exact full-name match
    return index


def resolve_aliases(
    names: list[str], cfg: dict[str, Any], index: dict[str, str], cache_path: Path
) -> dict[str, str]:
    """Resolve MTGO-only printed names (e.g. 'Leyline Weaver' -> 'Spider Manifestation').

    Uses Scryfall's /cards/named fuzzy endpoint, accepting a result only if the query
    equals the name, printed_name or flavor_name of the card or of one of its faces.
    Results are cached in ``data/raw/scryfall_aliases.json``.
    """
    cache: dict[str, str | None] = {}
    if cache_path.exists():
        cache = json.loads(cache_path.read_text(encoding="utf-8"))
    src = cfg["source"]
    headers = {"User-Agent": src["user_agent"], "Accept": "application/json"}
    for name in names:
        if cache.get(name):
            continue  # cached hit; misses are retried (only a handful of names)
        time.sleep(float(src["request_delay_s"]))
        try:
            resp = requests.get(src["scryfall_named_api"], params={"fuzzy": name}, headers=headers, timeout=20)
        except requests.RequestException:
            continue  # offline: leave unresolved, do not cache
        data = resp.json() if resp.ok else {}
        keys = ("name", "printed_name", "flavor_name")
        objs = [data, *data.get("card_faces", [])]  # MTGO names of DFCs live on the faces
        candidates = {str(o.get(k, "")).lower() for o in objs for k in keys}
        cache[name] = data.get("name") if name.lower() in candidates else None
    cache_path.write_text(json.dumps(cache, indent=1, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    return {k: index[v.lower()] for k, v in cache.items() if v and v.lower() in index}


# --------------------------------------------------------------------------- decks
def iter_tournament_files(repo: Path, cfg: dict[str, Any]) -> Iterator[Path]:
    """Yield every tournament JSON of the configured sources, in a deterministic order."""
    for source in cfg["source"]["deck_sources"]:
        yield from sorted((repo / "Tournaments" / source).rglob("*.json"))


def read_raw_decks(repo: Path, cfg: dict[str, Any]) -> list[dict[str, Any]]:
    """Read every deck of the configured format. Player names are dropped here."""
    fmt, use_sb = cfg["data"]["format"], cfg["data"]["use_sideboard"]
    decks = []
    for path in iter_tournament_files(repo, cfg):
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        tour = data.get("Tournament", {})
        if tour.get("Formats") != fmt:
            continue
        for deck in data.get("Decks") or []:
            cards = list(deck.get("Mainboard") or [])
            if use_sb:
                cards += deck.get("Sideboard") or []
            decks.append({
                "date": tour["Date"][:10],
                "event": tour.get("Name", ""),
                "result": deck.get("Result", ""),
                "cards": [(c["CardName"], int(c["Count"])) for c in cards],
            })
    return decks


def filter_window(decks: list[dict[str, Any]], cfg: dict[str, Any]) -> tuple[list[dict[str, Any]], str, str]:
    """Keep decks in (end - months_back, end]; end defaults to the latest date available."""
    end = pd.Timestamp(cfg["data"]["end_date"] or max(d["date"] for d in decks))
    start = end - pd.DateOffset(months=int(cfg["data"]["months_back"]))
    kept = [d for d in decks if start < pd.Timestamp(d["date"]) <= end]
    return kept, str(start.date()), str(end.date())


def load_decks(
    repo: str | Path, cfg: dict[str, Any], cards: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], pd.DataFrame]:
    """Parse, clean and filter decks.

    Returns ``(decks_long, deck_meta, report, unmatched)`` where ``decks_long`` has columns
    ``deck_id, date, card_name, copies`` (no basic lands, no player names).
    """
    repo = Path(repo)
    index = build_name_index(cards)
    basic_names = set(cards.loc[cards["type_line"].map(is_basic_land), "name"])

    raw = read_raw_decks(repo, cfg)
    decks, start, end = filter_window(raw, cfg)
    report: dict[str, Any] = {"decks_in_format_total": len(raw), "window_start_exclusive": start,
                              "window_end": end, "decks_in_window": len(decks)}

    # Canonicalize names (exact / front face / 'A/B' / MTGO aliases).
    occurrences = Counter(normalize_deck_name(n) for d in decks for n, _ in d["cards"])
    unresolved = sorted(n for n in occurrences if n.lower() not in index)
    aliases = resolve_aliases(unresolved, cfg, index, Path(cfg["paths"]["raw"]) / "scryfall_aliases.json")
    canonical = {n: aliases.get(n) or index.get(n.lower()) for n in occurrences}
    report["aliases_resolved"] = {k: v for k, v in aliases.items() if k in occurrences}

    unmatched = Counter()
    min_main = int(cfg["data"]["min_main_cards"])
    seen: set[tuple] = set()
    n_small = n_dup = 0
    rows, meta = [], []
    for d in sorted(decks, key=lambda x: x["date"]):
        counts: Counter = Counter()
        for name, cnt in d["cards"]:
            norm = normalize_deck_name(name)
            if canonical[norm] is None:
                unmatched[norm] += cnt
            counts[canonical[norm] or f"?{norm}"] += cnt
        if sum(counts.values()) < min_main:
            n_small += 1
            continue
        signature = tuple(sorted(counts.items()))
        if cfg["data"]["drop_duplicate_decks"] and signature in seen:
            n_dup += 1
            continue
        seen.add(signature)
        deck_id = len(meta)
        meta.append({"deck_id": deck_id, "date": d["date"], "event": d["event"],
                     "result": d["result"], "n_main_cards": sum(counts.values())})
        for name, cnt in sorted(counts.items()):
            if name.startswith("?") or name in basic_names:
                continue
            rows.append({"deck_id": deck_id, "date": d["date"], "card_name": name, "copies": cnt})

    decks_long = pd.DataFrame(rows)
    deck_meta = pd.DataFrame(meta)
    total_copies = sum(cnt for d in decks for _, cnt in d["cards"])
    unmatched_df = pd.DataFrame(
        sorted(unmatched.items(), key=lambda x: -x[1]), columns=["card_name", "copies_in_window"]
    )
    report.update({
        "decks_removed_small": n_small,
        "decks_removed_duplicate": n_dup,
        "decks_kept": len(deck_meta),
        "distinct_names_in_window": len(occurrences),
        "unmatched_distinct": len(unmatched),
        "unmatched_distinct_pct": 100 * len(unmatched) / max(len(occurrences), 1),
        "unmatched_copies_pct": 100 * sum(unmatched.values()) / max(total_copies, 1),
    })
    return decks_long, deck_meta, report, unmatched_df


# --------------------------------------------------------------------------- catalog
def build_catalog(cards: pd.DataFrame, decks: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Catalog = non-basic cards appearing in at least one deck, sorted by name."""
    used = sorted(decks["card_name"].unique())
    catalog = cards.set_index("name").loc[used].reset_index()
    catalog.insert(0, "item_idx", range(len(catalog)))
    return catalog, dict(zip(catalog["name"], catalog["item_idx"]))


def main() -> None:
    """Run the full load step and print the data report."""
    cfg = load_config()
    raw, processed, results = (Path(cfg["paths"][k]) for k in ("raw", "processed", "results"))

    cards = load_cards(raw / "oracle_cards.jsonl.gz", cfg["data"]["format"])
    decks, meta, report, unmatched = load_decks(raw / "MTGODecklistCache", cfg, cards)
    catalog, _ = build_catalog(cards, decks)

    report.update({
        "scryfall_cards": len(cards),
        "catalog_items": len(catalog),
        "interactions": len(decks),
        "sparsity_pct": 100 * (1 - len(decks) / (len(meta) * len(catalog))),
        "mean_distinct_nonbasic_per_deck": float(decks.groupby("deck_id").size().mean()),
    })
    ok = report["decks_kept"] >= cfg["data"]["min_decks"] and len(catalog) >= cfg["data"]["min_items"]
    report["meets_minimums"] = bool(ok)

    cards.to_parquet(processed / "cards.parquet", index=False)
    decks.to_parquet(processed / "decks.parquet", index=False)
    meta.to_parquet(processed / "deck_meta.parquet", index=False)
    catalog.to_parquet(processed / "catalog.parquet", index=False)
    unmatched.to_csv(results / "unmatched_cards.csv", index=False)
    (processed / "load_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not ok:
        raise SystemExit("Mínimos não atingidos (decks/itens): amplie months_back no config.yaml.")


if __name__ == "__main__":
    main()
