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


MAX_TEST = 64  # test slice is reserved at this size so eval never overlaps it


def _split(xs: list[str], n_train: int, n_test: int, n_eval: int, seed: int):
    xs = xs[:]
    random.Random(seed).shuffle(xs)
    eval_start = n_train + MAX_TEST
    return xs[:n_train], xs[n_train : n_train + n_test], xs[eval_start : eval_start + n_eval]


def load_splits(
    n_train: int = 128, n_test: int = 32, n_eval: int = 0, seed: int = 0
) -> dict[str, list[str]]:
    n_test = min(n_test, MAX_TEST)
    out = {}
    for name, xs in (("harmful", harmful()), ("harmless", harmless())):
        out[f"{name}_train"], out[f"{name}_test"], out[f"{name}_eval"] = _split(
            xs, n_train, n_test, n_eval, seed
        )
    return out
