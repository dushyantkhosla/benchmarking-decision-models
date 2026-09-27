from core.metrics import accuracy, latency_stats, macro_f1


def test_accuracy():
    assert accuracy(["a", "b", "c"], ["a", "x", "c"]) == 2 / 3
    assert accuracy([], []) == 0.0


def test_macro_f1_perfect():
    assert macro_f1(["a", "b"], ["a", "b"]) == 1.0


def test_macro_f1_partial():
    # labels: a, b. predictions swap one 'a' for 'b'.
    golds = ["a", "a", "b"]
    preds = ["a", "b", "b"]
    # a: tp=1 fp=0 fn=1 -> f1=2/3 ; b: tp=1 fp=1 fn=0 -> f1=2/3
    assert abs(macro_f1(golds, preds) - 2 / 3) < 1e-9


def test_latency_stats():
    stats = latency_stats([10.0, 20.0, 30.0, 40.0])
    assert stats["count"] == 4
    assert stats["mean"] == 25.0
    assert stats["min"] == 10.0
    assert stats["max"] == 40.0
    assert stats["p50"] in (20.0, 30.0)
