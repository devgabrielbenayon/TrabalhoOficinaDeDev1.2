"""Download raw data: Scryfall oracle cards bulk and the MTGO decklist cache.

Idempotent: files that already exist are not downloaded again.

Usage: python -m data.download
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any

import requests

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config

ORACLE_FILENAME = "oracle_cards.jsonl.gz"
DECKS_DIRNAME = "MTGODecklistCache"


def scryfall_headers(cfg: dict[str, Any]) -> dict[str, str]:
    """HTTP headers required by the Scryfall API guidelines."""
    return {"User-Agent": cfg["source"]["user_agent"], "Accept": "application/json;q=0.9,*/*;q=0.8"}


def download_scryfall(cfg: dict[str, Any]) -> Path:
    """Download the Scryfall bulk file (JSONL, gzip) unless it is already cached."""
    out = Path(cfg["paths"]["raw"]) / ORACLE_FILENAME
    if out.exists():
        print(f"[scryfall] já existe: {out.name} (pulando download)")
        return out

    src = cfg["source"]
    resp = requests.get(src["scryfall_bulk_api"], headers=scryfall_headers(cfg), timeout=30)
    resp.raise_for_status()
    entry = next(x for x in resp.json()["data"] if x["type"] == src["scryfall_bulk_type"])
    url = entry.get("jsonl_download_uri") or entry["download_uri"]
    print(f"[scryfall] baixando {entry['name']} ({entry.get('updated_at')}) de {url}")

    tmp = out.with_suffix(".part")
    with requests.get(url, headers=scryfall_headers(cfg), stream=True, timeout=120) as r:
        r.raise_for_status()
        with open(tmp, "wb") as fh:
            for chunk in r.iter_content(chunk_size=1 << 20):
                fh.write(chunk)
    tmp.rename(out)
    (out.parent / "oracle_cards.version").write_text(str(entry.get("updated_at")), encoding="utf-8")
    print(f"[scryfall] ok: {out.stat().st_size / 1e6:.1f} MB")
    return out


def _git(repo: Path, *args: str) -> str:
    """Run a git command inside ``repo`` and return stdout."""
    res = subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, text=True)
    return res.stdout.strip()


def _month_folders(repo: Path, source: str, n_months: int) -> list[str]:
    """Return the last ``n_months`` ``Tournaments/<source>/YYYY/MM`` folders in the pinned tree."""
    root = f"Tournaments/{source}"
    years = _git(repo, "ls-tree", "--name-only", "HEAD", f"{root}/").splitlines()
    months: list[str] = []
    for year in sorted(years):
        months += _git(repo, "ls-tree", "--name-only", "HEAD", f"{year}/").splitlines()
    return sorted(months)[-n_months:]


def download_decks(cfg: dict[str, Any]) -> Path:
    """Sparse, blob-less checkout of only the month folders needed for the time window."""
    src = cfg["source"]
    repo = Path(cfg["paths"]["raw"]) / DECKS_DIRNAME
    if not (repo / ".git").exists():
        print(f"[decks] clonando (esparso, sem blobs) {src['decks_repo']}")
        repo.mkdir(parents=True, exist_ok=True)
        _git(repo, "init", "-q")
        _git(repo, "remote", "add", "origin", src["decks_repo"])
        _git(repo, "config", "remote.origin.promisor", "true")
        _git(repo, "config", "remote.origin.partialclonefilter", "blob:none")

    try:
        head = _git(repo, "rev-parse", "HEAD")
    except subprocess.CalledProcessError:
        head = ""
    if head != src["decks_commit"]:
        print(f"[decks] buscando commit fixado {src['decks_commit'][:10]}")
        _git(repo, "fetch", "-q", "--depth", "1", "--filter=blob:none", "origin", src["decks_commit"])
        _git(repo, "sparse-checkout", "set", "--no-cone", "/README.md")
        _git(repo, "checkout", "-q", "--detach", src["decks_commit"])

    # One extra month so the window [end - months_back, end] is always fully covered.
    n_months = int(cfg["data"]["months_back"]) + 1
    folders = [f for s in src["deck_sources"] for f in _month_folders(repo, s, n_months)]
    print(f"[decks] pastas no recorte: {', '.join(folders)}")
    _git(repo, "sparse-checkout", "set", "--no-cone", *[f"/{f}/" for f in folders])
    n_files = sum(1 for f in folders for _ in (repo / f).rglob("*.json"))
    print(f"[decks] ok: {n_files} arquivos de torneio")
    return repo


def main() -> None:
    """Download everything needed by the pipeline."""
    cfg = load_config()
    download_scryfall(cfg)
    download_decks(cfg)


if __name__ == "__main__":
    main()
