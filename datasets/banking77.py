"""BANKING77 dataset adapter.

Source of record: PolyAI-LDN/task-specific-datasets (CC-BY-4.0).
The Hugging Face repo (PolyAI/banking77) wraps these exact CSVs via a
legacy loader script, so we fetch the CSVs directly from GitHub.

Schema: CSV with columns `text,category` where category is one of 77
string intent labels.
"""

from __future__ import annotations

import csv
import hashlib
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import httpx

from core.schemas import DecisionExample

TRAIN_URL = "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/train.csv"
TEST_URL = "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/test.csv"

TRACK = "classification"
DATASET = "banking77"
INSTRUCTIONS = "Classify the banking customer query into exactly one intent category."

TIERS = ("reduced", "full")


@dataclass
class Row:
    index: int  # row number within the split file (stable identifier)
    text: str
    category: str


def download(cache_dir: Path, split: str = "test") -> tuple[Path, str]:
    """Fetch the split CSV into cache_dir. Returns (path, sha256_hex)."""
    url = {"train": TRAIN_URL, "test": TEST_URL}[split]
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{split}.csv"
    if not path.exists():
        resp = httpx.get(url, follow_redirects=True, timeout=60.0)
        resp.raise_for_status()
        path.write_bytes(resp.content)
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


def load_rows(path: Path) -> list[Row]:
    rows: list[Row] = []
    with path.open(newline="", encoding="utf-8") as f:
        for i, rec in enumerate(csv.DictReader(f)):
            rows.append(Row(index=i, text=rec["text"], category=rec["category"]))
    return rows


def stratified_sample(rows: list[Row], n: int, seed: int) -> list[Row]:
    """Deterministic label-balanced sample via seeded round-robin."""
    by_label: dict[str, list[Row]] = defaultdict(list)
    for row in rows:
        by_label[row.category].append(row)
    rng = random.Random(seed)
    for group in by_label.values():
        rng.shuffle(group)
    labels = sorted(by_label)
    picked: list[Row] = []
    depth = 0
    while len(picked) < min(n, len(rows)):
        for label in labels:
            if depth < len(by_label[label]):
                picked.append(by_label[label][depth])
                if len(picked) >= n:
                    return picked
        depth += 1
    return picked


def build_examples(
    rows: list[Row],
    labels: list[str],
    tier: str,
    seed: int,
    split: str = "test",
    n_distractors: int = 9,
) -> list[DecisionExample]:
    """Build normalized examples for one difficulty tier.

    reduced: gold + n_distractors sampled with an RNG seeded by
             (seed, example id) — deterministic and order-independent.
    full:    every label.
    """
    examples: list[DecisionExample] = []
    for row in rows:
        # Tier is part of the id: different option sets are different decisions.
        example_id = f"{DATASET}:{split}:{row.index}:{tier}"
        if tier == "reduced":
            rng = random.Random(f"{seed}:{example_id}")
            others = [lab for lab in labels if lab != row.category]
            distractors = rng.sample(others, min(n_distractors, len(others)))
            options = [row.category, *distractors]
            rng.shuffle(options)
        elif tier == "full":
            options = list(labels)
        else:
            raise ValueError(f"unknown tier: {tier}")
        examples.append(
            DecisionExample(
                id=example_id,
                track=TRACK,
                dataset=DATASET,
                input={"text": row.text},
                decision={
                    "type": "choice",
                    "question": "intent",
                    "instructions": INSTRUCTIONS,
                    "options": options,
                },
                gold=row.category,
                metadata={"difficulty": f"{tier}-choice", "split": split},
            )
        )
    return examples


def load_labels(rows: list[Row]) -> list[str]:
    return sorted({row.category for row in rows})
