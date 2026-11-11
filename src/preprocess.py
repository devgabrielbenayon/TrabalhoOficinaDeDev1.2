"""Text cleaning of rules text and normalization of numeric card attributes."""

from __future__ import annotations

import re
from functools import lru_cache

import nltk
import numpy as np
import pandas as pd
from nltk.stem.snowball import SnowballStemmer

CARDNAME_TOKEN = "cardname"
_REMINDER_RE = re.compile(r"\([^)]*\)")
_SYMBOL_RE = re.compile(r"\{([^}]+)\}")
_PT_MOD_RE = re.compile(r"([+-])[\dX]+/([+-])[\dX]+")
_NON_WORD_RE = re.compile(r"[^a-z0-9_]+")
_NUMBER_RE = re.compile(r"^-?\d+(\.\d+)?$")

_SINGLE_SYMBOLS = {
    "T": "tap", "Q": "untap", "E": "energy", "S": "mana_snow", "C": "mana_c",
    "X": "mana_x", "Y": "mana_x", "Z": "mana_x", "P": "mana_phyrexian",
    "W": "mana_w", "U": "mana_u", "B": "mana_b", "R": "mana_r", "G": "mana_g",
}


@lru_cache(maxsize=1)
def english_stopwords() -> frozenset[str]:
    """NLTK English stopwords, downloaded automatically on first use."""
    try:
        words = nltk.corpus.stopwords.words("english")
    except LookupError:
        nltk.download("stopwords", quiet=True)
        words = nltk.corpus.stopwords.words("english")
    return frozenset(words)


_STEMMER = SnowballStemmer("english")


# --------------------------------------------------------------------------- text
def card_name_variants(name: str, type_line: str = "") -> list[str]:
    """Names by which a card refers to itself: full name, face names and, for
    legendary cards, the short name before the comma ("Ragavan")."""
    variants = {name, *name.split(" // ")}
    if "Legendary" in type_line:
        for face in name.split(" // "):
            short = face.split(",")[0].strip()
            if len(short) >= 3:
                variants.add(short)
    return sorted((v for v in variants if v), key=len, reverse=True)


def replace_card_name(text: str, name: str, type_line: str = "") -> str:
    """Step 1: replace self-references with the token ``cardname``."""
    for variant in card_name_variants(name, type_line):
        text = re.sub(rf"\b{re.escape(variant)}\b", CARDNAME_TOKEN, text)
    return text


def remove_reminder_text(text: str) -> str:
    """Step 2: drop reminder text between parentheses."""
    return _REMINDER_RE.sub(" ", text)


def _symbol_to_tokens(symbol: str) -> str:
    """Translate the content of one ``{...}`` symbol into word tokens."""
    tokens = []
    for part in symbol.upper().split("/"):  # hybrid {W/U}, {2/W}, phyrexian {G/P}
        if part.isdigit() or part == "½":
            tokens.append("mana_generic")
        elif part in _SINGLE_SYMBOLS:
            tokens.append(_SINGLE_SYMBOLS[part])
        else:
            tokens.append("sym_" + re.sub(r"\W", "", part.lower()))
    return " ".join(tokens)


def convert_symbols(text: str) -> str:
    """Step 3: mana/cost symbols to tokens, plus +N/+N and -N/-N modifiers."""
    text = _SYMBOL_RE.sub(lambda m: f" {_symbol_to_tokens(m.group(1))} ", text)

    def pt_mod(m: re.Match) -> str:
        signs = m.group(1) + m.group(2)
        return {"++": " pt_plus ", "--": " pt_minus "}.get(signs, " pt_mixed ")

    return _PT_MOD_RE.sub(pt_mod, text)


def normalize_tokens(text: str) -> str:
    """Step 4: lower-case, strip punctuation, remove stopwords and stem."""
    stop = english_stopwords()
    tokens = []
    for tok in _NON_WORD_RE.sub(" ", text.lower()).split():
        if tok in stop:
            continue
        # Tokens we created ourselves (mana_r, cardname...) are kept verbatim.
        tokens.append(tok if "_" in tok or tok == CARDNAME_TOKEN else _STEMMER.stem(tok))
    return " ".join(tokens)


def clean_text(text: str, name: str, type_line: str = "") -> str:
    """Full rules-text cleaning pipeline (steps 1-4, in the specified order)."""
    text = replace_card_name(text or "", name, type_line)
    text = remove_reminder_text(text)
    text = convert_symbols(text)
    return normalize_tokens(text)


# --------------------------------------------------------------------------- numeric
def parse_pt(value: object) -> tuple[float, int]:
    """Power/toughness to ``(number, variable_flag)``; '*', '1+*' -> (0, 1); missing -> (0, 0)."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return 0.0, 0
    s = str(value).strip()
    if _NUMBER_RE.match(s):
        return float(s), 0
    return 0.0, 1


def minmax(values: pd.Series) -> pd.Series:
    """Min-max scale to [0, 1]; constant columns become 0."""
    lo, hi = values.min(), values.max()
    return (values - lo) / (hi - lo) if hi > lo else values * 0.0


def preprocess_cards(catalog: pd.DataFrame) -> pd.DataFrame:
    """Add ``clean_text`` and normalized numeric columns to the catalog (missing values handled)."""
    df = catalog.copy()
    df["oracle_text"] = df["oracle_text"].fillna("")
    df["type_line"] = df["type_line"].fillna("")
    df["clean_text"] = [
        clean_text(t, n, tl) for t, n, tl in zip(df["oracle_text"], df["name"], df["type_line"])
    ]

    power = df["power"].map(parse_pt)
    tough = df["toughness"].map(parse_pt)
    df["is_creature"] = df["type_line"].str.contains(r"\bCreature\b").astype(int)
    df["variable_pt"] = [max(p[1], t[1]) for p, t in zip(power, tough)]
    # Non-creatures without P/T get 0; Vehicles keep their printed P/T (it is meaningful).
    df["num_cmc"] = minmax(df["cmc"].fillna(0.0).astype(float))
    df["num_power"] = minmax(power.map(lambda p: p[0]).clip(lower=0))
    df["num_toughness"] = minmax(tough.map(lambda t: t[0]).clip(lower=0))
    return df
