"""Normalized, model-independent benchmark schemas.

DecisionExample is the canonical input unit produced by dataset adapters.
Prediction is the canonical output unit produced by model adapters.
Both serialize to plain dicts for immutable JSON/JSONL result artifacts.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable


@dataclass
class DecisionExample:
    id: str
    track: str
    dataset: str
    input: dict[str, Any]
    decision: dict[str, Any]  # {"type": "choice", "question": ..., "instructions": ..., "options": [...]}
    gold: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Prediction:
    example_id: str
    model: str
    answer: str | None
    confidence: float | None = None
    probabilities: dict[str, float] | None = None
    latency_ms: float | None = None  # wall-clock, end to end
    eval_ms: float | None = None  # model-reported eval duration
    input_tokens: int | None = None
    state_truncated: bool = False
    error: str | None = None
    raw: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def write_jsonl(path: Path, records: Iterable[dict[str, Any]], append: bool = False) -> None:
    mode = "a" if append else "w"
    with path.open(mode, encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
