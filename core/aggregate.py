"""Recompute aggregate metrics from a run's raw artifacts.

Reads results/<run_id>/predictions.jsonl + examples.jsonl and writes
metrics_from_artifacts.json. Read-only with respect to predictions and
examples; never touches the incremental metrics.json.

Output shape:
    {"cells": {"<model> | <tier>": {...}},
     "by_criterion": {"<model> | <criterion>": {...}}}   # binary tracks only

Usage:
    uv run python -m core.aggregate results/<run_id>
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

from core.metrics import accuracy, brier_score, latency_stats, macro_f1, precision_recall_f1


def _cell_stats(c: dict) -> dict:
    stats = {
        "n": len(c["golds"]) + c["errors"],
        "errors": c["errors"],
        "truncated": c["truncated"],
        "accuracy": accuracy(c["golds"], c["answers"]),
        "macro_f1": macro_f1(c["golds"], c["answers"]),
        "latency_ms": latency_stats(c["latencies"]),
    }
    if c["dtype"] == "noul" and c["golds"]:
        stats.update(precision_recall_f1(c["golds"], c["answers"]))
        stats["brier"] = brier_score(c["golds"], c["probs_yes"])
    return stats


def _new_cell(dtype: str | None = None) -> dict:
    return {
        "golds": [], "answers": [], "latencies": [], "probs_yes": [],
        "errors": 0, "truncated": 0, "dtype": dtype,
    }


def aggregate(run_dir: Path) -> dict:
    meta: dict[str, dict] = {}
    for line in (run_dir / "examples.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        e = json.loads(line)
        meta[e["id"]] = {
            "gold": e["gold"],
            "tier": e["metadata"]["difficulty"],
            "criterion": e["metadata"].get("criterion"),
            "dtype": e["decision"].get("type"),
        }

    cells: dict[tuple[str, str], dict] = defaultdict(_new_cell)
    criteria: dict[tuple[str, str], dict] = defaultdict(_new_cell)
    for line in (run_dir / "predictions.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            p = json.loads(line)
        except json.JSONDecodeError:
            continue  # a partially written final line while a run is live
        m = meta.get(p["example_id"])
        if m is None:
            continue
        targets = [cells[(p["model"], m["tier"])]]
        if m["criterion"]:
            targets.append(criteria[(p["model"], m["criterion"])])
        for c in targets:
            c["dtype"] = m["dtype"]
            if p["error"] is None:
                c["golds"].append(m["gold"])
                c["answers"].append(p["answer"])
                if m["dtype"] == "noul" and p["probabilities"]:
                    c["probs_yes"].append(p["probabilities"]["yes"])
            else:
                c["errors"] += 1
            if p["state_truncated"]:
                c["truncated"] += 1
            if p["latency_ms"] is not None:
                c["latencies"].append(p["latency_ms"])

    return {
        "cells": {f"{model} | {tier}": _cell_stats(c) for (model, tier), c in sorted(cells.items())},
        "by_criterion": {
            f"{model} | {crit}": _cell_stats(c) for (model, crit), c in sorted(criteria.items())
        },
    }


def _print(key: str, v: dict) -> None:
    line = (
        f"{key}: n={v['n']} acc={v['accuracy']:.3f} mF1={v['macro_f1']:.3f} "
        f"p50={v['latency_ms']['p50']:.0f}ms errors={v['errors']}"
    )
    if "f1" in v:
        line += f" | P={v['precision']:.3f} R={v['recall']:.3f} F1={v['f1']:.3f} Brier={v['brier']:.3f}"
    print(line)


if __name__ == "__main__":
    run_dir = Path(sys.argv[1])
    result = aggregate(run_dir)
    out_path = run_dir / "metrics_from_artifacts.json"
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {out_path}")
    print("-- cells")
    for key, v in result["cells"].items():
        _print(key, v)
    if result["by_criterion"]:
        print("-- by criterion")
        for key, v in result["by_criterion"].items():
            _print(key, v)
