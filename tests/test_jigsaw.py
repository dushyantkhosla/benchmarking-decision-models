from core.metrics import brier_score, precision_recall_f1
from datasets.jigsaw import CRITERIA, Row, build_examples, sample_pairs


def make_rows() -> list[Row]:
    rows = []
    for i in range(100):
        # rows 0-9 positive for every criterion; rest negative
        labels = {c: (1 if i < 10 else 0) for c in CRITERIA}
        rows.append(Row(index=i, text=f"comment {i}", labels=labels))
    return rows


def test_sample_pairs_deterministic_and_balanced():
    rows = make_rows()
    a = sample_pairs(rows, n_pos=5, n_neg=7, seed=42)
    b = sample_pairs(rows, n_pos=5, n_neg=7, seed=42)
    assert [(r.index, c) for r, c in a] == [(r.index, c) for r, c in b]
    assert len(a) == len(CRITERIA) * 12
    for criterion in CRITERIA:
        pairs = [(r, c) for r, c in a if c == criterion]
        assert sum(1 for r, _ in pairs if r.labels[criterion] == 1) == 5
        assert sum(1 for r, _ in pairs if r.labels[criterion] == 0) == 7


def test_build_examples_shape():
    rows = make_rows()
    pairs = sample_pairs(rows, n_pos=2, n_neg=2, seed=1)
    examples = build_examples(pairs)
    assert len(examples) == len(CRITERIA) * 4
    for ex in examples:
        assert ex.decision["type"] == "noul"
        assert ex.gold in ("yes", "no")
        assert ex.metadata["criterion"] in CRITERIA
        assert ex.id.endswith(ex.metadata["criterion"])


def test_precision_recall_f1():
    golds = ["yes", "yes", "no", "no"]
    preds = ["yes", "no", "yes", "no"]
    m = precision_recall_f1(golds, preds)
    assert m["precision"] == 0.5
    assert m["recall"] == 0.5
    assert m["f1"] == 0.5


def test_brier_score():
    golds = ["yes", "no"]
    probs = [0.9, 0.1]
    assert abs(brier_score(golds, probs) - 0.01) < 1e-9
