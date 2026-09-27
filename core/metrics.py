"""Benchmark metrics. No third-party dependencies."""

from __future__ import annotations

from typing import Sequence


def accuracy(golds: Sequence[str], preds: Sequence[str]) -> float:
    if not golds:
        return 0.0
    correct = sum(1 for g, p in zip(golds, preds) if g == p)
    return correct / len(golds)


def macro_f1(golds: Sequence[str], preds: Sequence[str]) -> float:
    """Macro-averaged F1 over the labels present in the gold set."""
    labels = sorted(set(golds))
    if not labels:
        return 0.0
    f1s = []
    for label in labels:
        tp = sum(1 for g, p in zip(golds, preds) if g == label and p == label)
        fp = sum(1 for g, p in zip(golds, preds) if g != label and p == label)
        fn = sum(1 for g, p in zip(golds, preds) if g == label and p != label)
        denom = 2 * tp + fp + fn
        f1s.append(2 * tp / denom if denom else 0.0)
    return sum(f1s) / len(f1s)


def latency_stats(values_ms: Sequence[float]) -> dict[str, float]:
    if not values_ms:
        return {"count": 0, "mean": 0.0, "p50": 0.0, "p95": 0.0, "min": 0.0, "max": 0.0}
    s = sorted(values_ms)

    def pct(q: float) -> float:
        # nearest-rank style percentile on sorted data
        idx = min(len(s) - 1, max(0, round(q * (len(s) - 1))))
        return s[idx]

    return {
        "count": len(s),
        "mean": sum(s) / len(s),
        "p50": pct(0.50),
        "p95": pct(0.95),
        "min": s[0],
        "max": s[-1],
    }


def precision_recall_f1(
    golds: Sequence[str], preds: Sequence[str], positive: str = "yes"
) -> dict[str, float]:
    tp = sum(1 for g, p in zip(golds, preds) if g == positive and p == positive)
    fp = sum(1 for g, p in zip(golds, preds) if g != positive and p == positive)
    fn = sum(1 for g, p in zip(golds, preds) if g == positive and p != positive)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def brier_score(
    golds: Sequence[str], probs_yes: Sequence[float], positive: str = "yes"
) -> float:
    """Mean squared error of P(positive) against the 0/1 outcome."""
    if not golds:
        return 0.0
    total = sum((p - (1.0 if g == positive else 0.0)) ** 2 for g, p in zip(golds, probs_yes))
    return total / len(golds)
