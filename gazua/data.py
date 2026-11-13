from __future__ import annotations

import csv
import json
import random
import urllib.request
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

ADVBENCH_URL = (
    "https://raw.githubusercontent.com/llm-attacks/llm-attacks/main/"
    "data/advbench/harmful_behaviors.csv"
)
ALPACA_URL = (
    "https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca/main/alpaca_data.json"
)


def _download(url: str, dest: Path) -> Path:
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        print(f"downloading {url}")
        urllib.request.urlretrieve(url, dest)
    return dest


def harmful() -> list[str]:
    path = _download(ADVBENCH_URL, DATA_DIR / "advbench_harmful_behaviors.csv")
    with open(path, newline="", encoding="utf-8") as f:
        return [row["goal"] for row in csv.DictReader(f)]


def harmless() -> list[str]:
    path = _download(ALPACA_URL, DATA_DIR / "alpaca_data.json")
    with open(path, encoding="utf-8") as f:
        rows = json.load(f)
    return [r["instruction"] for r in rows if not r["input"].strip()]


def _split(xs: list[str], n_train: int, n_test: int, seed: int):
    xs = xs[:]
    random.Random(seed).shuffle(xs)
    return xs[:n_train], xs[n_train : n_train + n_test]


def load_splits(n_train: int = 128, n_test: int = 32, seed: int = 0) -> dict[str, list[str]]:
    h_train, h_test = _split(harmful(), n_train, n_test, seed)
    s_train, s_test = _split(harmless(), n_train, n_test, seed)
    return {
        "harmful_train": h_train,
        "harmful_test": h_test,
        "harmless_train": s_train,
        "harmless_test": s_test,
    }
