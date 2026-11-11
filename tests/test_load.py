"""Unit tests for card-name handling and basic-land detection."""

import pandas as pd

from src.load import build_catalog, build_name_index, is_basic_land, normalize_deck_name


def test_basic_land_detection() -> None:
    assert is_basic_land("Basic Land — Forest")
    assert is_basic_land("Basic Snow Land — Island")  # Snow-Covered basics
    assert is_basic_land("Basic Land")  # Wastes
    assert not is_basic_land("Land — Forest Island")  # Breeding Pool style duals
    assert not is_basic_land("Basic Creature — Shapeshifter")
    assert not is_basic_land("Legendary Land")


def test_normalize_split_names() -> None:
    assert normalize_deck_name("Fire/Ice") == "Fire // Ice"
    assert normalize_deck_name("Fire // Ice") == "Fire // Ice"
    assert normalize_deck_name("  Lightning  Bolt ") == "Lightning Bolt"


def test_name_index_matches_full_and_front_face() -> None:
    cards = pd.DataFrame({
        "name": ["Fable of the Mirror-Breaker // Reflection of Kiki-Jiki", "Fire // Ice", "Fire"],
        "front_name": ["Fable of the Mirror-Breaker", "Fire", "Fire"],
    })
    idx = build_name_index(cards)
    assert idx["fable of the mirror-breaker"] == "Fable of the Mirror-Breaker // Reflection of Kiki-Jiki"
    assert idx["fire // ice"] == "Fire // Ice"
    assert idx["fire"] == "Fire"  # an exact full-name match is never overridden by a front face


def test_catalog_only_contains_used_cards() -> None:
    cards = pd.DataFrame({"name": ["A", "B", "C"], "front_name": ["A", "B", "C"]})
    decks = pd.DataFrame({"deck_id": [0, 0, 1], "card_name": ["C", "A", "C"], "copies": [1, 4, 2]})
    catalog, mapping = build_catalog(cards, decks)
    assert list(catalog["name"]) == ["A", "C"]
    assert mapping == {"A": 0, "C": 1}
