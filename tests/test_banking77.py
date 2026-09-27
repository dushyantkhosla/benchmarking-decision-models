from datasets.banking77 import Row, build_examples, load_labels, stratified_sample


def make_rows(n_per_label: int = 5, labels: int = 7) -> list[Row]:
    rows = []
    i = 0
    for lab in range(labels):
        for _ in range(n_per_label):
            rows.append(Row(index=i, text=f"text {i}", category=f"label_{lab}"))
            i += 1
    return rows


def test_stratified_sample_is_deterministic():
    rows = make_rows()
    a = stratified_sample(rows, n=20, seed=42)
    b = stratified_sample(rows, n=20, seed=42)
    assert [r.index for r in a] == [r.index for r in b]


def test_stratified_sample_is_balanced():
    rows = make_rows(n_per_label=5, labels=7)
    sample = stratified_sample(rows, n=21, seed=42)
    counts = {}
    for r in sample:
        counts[r.category] = counts.get(r.category, 0) + 1
    assert set(counts.values()) == {3}


def test_reduced_tier_deterministic_and_valid():
    rows = make_rows(labels=12)
    labels = load_labels(rows)
    a = build_examples(rows, labels, tier="reduced", seed=42)
    b = build_examples(rows, labels, tier="reduced", seed=42)
    assert [e.decision["options"] for e in a] == [e.decision["options"] for e in b]
    for ex in a:
        opts = ex.decision["options"]
        assert ex.gold in opts
        assert len(opts) == len(set(opts)) == 10


def test_full_tier_has_all_labels():
    rows = make_rows()
    labels = load_labels(rows)
    examples = build_examples(rows, labels, tier="full", seed=42)
    for ex in examples:
        assert ex.decision["options"] == labels
