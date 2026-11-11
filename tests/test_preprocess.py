"""Tests for rules-text cleaning and numeric parsing."""

import pandas as pd
import pytest

from src.preprocess import (
    clean_text, convert_symbols, minmax, parse_pt, preprocess_cards,
    remove_reminder_text, replace_card_name,
)


def test_replace_card_name_full_and_short_legendary() -> None:
    text = "Whenever Ragavan, Nimble Pilferer deals damage, Ragavan gets +1/+1."
    out = replace_card_name(text, "Ragavan, Nimble Pilferer", "Legendary Creature — Monkey Pirate")
    assert "Ragavan" not in out and out.count("cardname") == 2


def test_replace_card_name_faces() -> None:
    out = replace_card_name("Fire deals 2 damage. Ice taps.", "Fire // Ice")
    assert out == "cardname deals 2 damage. cardname taps."


def test_remove_reminder_text() -> None:
    assert remove_reminder_text("Flying (It can't be blocked.)").strip() == "Flying"


@pytest.mark.parametrize("raw, expected", [
    ("{R}", "mana_r"), ("{2}", "mana_generic"), ("{T}", "tap"), ("{X}", "mana_x"),
    ("{W/U}", "mana_w mana_u"), ("{2/B}", "mana_generic mana_b"), ("{G/P}", "mana_g mana_phyrexian"),
])
def test_convert_mana_symbols(raw: str, expected: str) -> None:
    assert convert_symbols(raw).split() == expected.split()


def test_convert_pt_modifiers() -> None:
    assert convert_symbols("gets +1/+1").split() == ["gets", "pt_plus"]
    assert convert_symbols("gets -2/-2").split() == ["gets", "pt_minus"]


def test_clean_text_full_pipeline() -> None:
    out = clean_text("Lightning Bolt deals 3 damage to any target. ({T}: nothing.)", "Lightning Bolt")
    assert out == "cardname deal 3 damag target"


def test_parse_pt() -> None:
    assert parse_pt("3") == (3.0, 0)
    assert parse_pt("*") == (0.0, 1)
    assert parse_pt("1+*") == (0.0, 1)
    assert parse_pt(None) == (0.0, 0)


def test_minmax_constant_column() -> None:
    assert minmax(pd.Series([2.0, 2.0])).tolist() == [0.0, 0.0]


def test_preprocess_cards_numeric_range() -> None:
    df = pd.DataFrame({
        "name": ["A", "B", "C"], "oracle_text": [None, "Draw a card.", ""],
        "type_line": ["Creature — Elf", "Instant", "Creature — Avatar"],
        "cmc": [1.0, 2.0, 5.0], "power": ["1", None, "*"], "toughness": ["1", None, "*"],
    })
    out = preprocess_cards(df)
    assert out["is_creature"].tolist() == [1, 0, 1]
    assert out["variable_pt"].tolist() == [0, 0, 1]
    for col in ["num_cmc", "num_power", "num_toughness"]:
        assert out[col].between(0, 1).all()
    assert out["clean_text"].tolist()[0] == ""
