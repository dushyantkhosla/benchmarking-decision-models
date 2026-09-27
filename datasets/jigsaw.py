"""Jigsaw Toxic Comment dataset adapter — binary adjudication track.

Source of record: the original Kaggle file set, mirrored at
thesofakillers/jigsaw-toxic-comment-classification-challenge (test.csv +
test_labels.csv). Labels are human annotations; the Kaggle representation is
ternary (-1 = unscored); we drop unscored rows, matching the HF mirror.

Schema: id, comment_text, and six binary criteria:
toxic, severe_toxic, obscene, threat, insult, identity_hate.

Each sampled (comment, criterion) pair becomes one yes/no decision:
"Does this text contain <criterion>?"
"""

from __future__ import annotations

import csv
import hashlib
import random
from dataclasses import dataclass
from pathlib import Path

import httpx

from core.schemas import DecisionExample

BASE_URL = "https://huggingface.co/datasets/thesofakillers/jigsaw-toxic-comment-classification-challenge/resolve/main"
FILES = {"test": "test.csv", "test_labels": "test_labels.csv"}

TRACK = "adjudication"
DATASET = "jigsaw"
CRITERIA = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

QUESTIONS = {
    "toxic": "Does this text contain toxicity?",
    "severe_toxic": "Does this text contain severe toxicity?",
    "obscene": "Does this text contain obscenity?",
    "threat": "Does this text contain a threat?",
    "insult": "Does this text contain an insult?",
    "identity_hate": "Does this text contain identity hate?",
}


@dataclass
class Row:
    index: int  # position in the scored set (stable identifier)
    text: str
    labels: dict[str, int]  # criterion -> 0/1


def download(cache_dir: Path) -> dict[str, tuple[Path, str]]:
    """Fetch both CSVs. Returns {name: (path, sha256_hex)}."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    out = {}
    for name, filename in FILES.items():
        path = cache_dir / filename
        if not path.exists():
            resp = httpx.get(f"{BASE_URL}/{filename}", follow_redirects=True, timeout=120.0)
            resp.raise_for_status()
            path.write_bytes(resp.content)
        out[name] = (path, hashlib.sha256(path.read_bytes()).hexdigest())
    return out


def load_rows(test_path: Path, labels_path: Path) -> list[Row]:
    """Merge test.csv + test_labels.csv, drop unscored (-1) rows."""
    texts: dict[str, str] = {}
    with test_path.open(newline="", encoding="utf-8") as f:
        for rec in csv.DictReader(f):
            texts[rec["id"]] = rec["comment_text"]

    rows: list[Row] = []
    with labels_path.open(newline="", encoding="utf-8") as f:
        for rec in csv.DictReader(f):
            if rec["toxic"] == "-1":  # unscored
                continue
            rows.append(
                Row(
                    index=len(rows),
                    text=texts[rec["id"]],
                    labels={c: int(rec[c]) for c in CRITERIA},
                )
            )
    return rows


def sample_pairs(
    rows: list[Row], n_pos: int, n_neg: int, seed: int
) -> list[tuple[Row, str]]:
    """Deterministic per-criterion balanced sample of (row, criterion) pairs."""
    pairs: list[tuple[Row, str]] = []
    for criterion in CRITERIA:
        pos = [r for r in rows if r.labels[criterion] == 1]
        neg = [r for r in rows if r.labels[criterion] == 0]
        rng = random.Random(f"{seed}:{criterion}")
        rng.shuffle(pos)
        rng.shuffle(neg)
        if len(pos) < n_pos:
            raise ValueError(f"{criterion}: only {len(pos)} positives, need {n_pos}")
        pairs.extend((r, criterion) for r in pos[:n_pos])
        pairs.extend((r, criterion) for r in neg[:n_neg])
    return pairs


def build_examples(pairs: list[tuple[Row, str]], split: str = "test") -> list[DecisionExample]:
    examples = []
    for row, criterion in pairs:
        examples.append(
            DecisionExample(
                id=f"{DATASET}:{split}:{row.index}:{criterion}",
                track=TRACK,
                dataset=DATASET,
                input={"text": row.text},
                decision={
                    "type": "noul",
                    "question": criterion,
                    "instructions": QUESTIONS[criterion],
                },
                gold="yes" if row.labels[criterion] == 1 else "no",
                metadata={"criterion": criterion, "difficulty": "binary", "split": split},
            )
        )
    return examples
