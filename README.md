# Decision-Model Benchmark

> When can a specialized decision model replace a general-purpose LLM call
> for binary adjudication, bounded classification, rubric scoring, and
> application routing?

This repository benchmarks small/fast decision models — **JEV, Laya, and
KEV** (including multiple KEV sizes where available) — on text-based
decision workloads. It is deliberately **not** a browser-automation or
general-agent benchmark.

## Tracks

Results are reported per track — never collapsed into one composite score.

| Track | Dataset | Decision type |
| --- | --- | --- |
| Binary adjudication | Jigsaw Toxic Comment | yes/no per criterion |
| Bounded multi-class classification | BANKING77 (reduced-choice + full 77) | one of N categories |
| Rubric scoring | NVIDIA HelpSteer2 | bounded ordinal score |
| Personal-finance routing | PersonalFinance_v2 (reviewed subset) | one route from query |
| Contextual finance routing | PF-Context (custom, curated) | case file + query → route |

## Principles

- **Zero-shot** — no fine-tuning of evaluated models.
- **Reproducible** — every run records dataset revision, exact example
  IDs, seed, model version, runtime, quantization/precision, and
  hardware; per-example predictions are persisted as immutable
  JSON/JSONL artifacts.
- **Operational metrics** — latency, throughput, memory, and
  invalid-output rates are measured alongside quality metrics
  (F1, macro-F1, ordinal MAE, calibration).
- **Confidence/fallback** — where confidence is available, report
  coverage-vs-error curves: how many decisions can stay local at a
  chosen error threshold.
- **Dataset hygiene** — license, provenance, schema, label definitions,
  imbalance, and contamination risk are verified and documented before
  any adapter is written.

## Layout

```text
datasets/     # dataset adapters (one per source dataset)
models/       # model adapters (one per model family; decide(example) -> prediction)
core/         # schemas, runner, metrics
results/      # immutable run artifacts
```

## Status

Vertical slice implemented: a reproducible, stratified BANKING77 subset
(`datasets/banking77.py`) → ollaya model adapter (`models/ollaya.py`) →
runner (`core/runner.py`) → persisted per-example predictions, accuracy,
macro-F1, and latency under `results/<run_id>/`.

- Stack: Python 3.13 + uv; deps: httpx only (pytest for dev).
- Run: `uv run python run_benchmark.py --models laya:en,kev:4b --tiers reduced,full --n 500`
- Tests: `uv run pytest`
- Dataset verification notes: `docs/dataset-verification.md`.

### Backends

Model specs route by suffix; runs record backend + runtime + precision per model.

Benchmark arms run on Apple GPU only. CPU-only runtimes are out of scope.

| Spec | Backend | Apple Silicon device |
| --- | --- | --- |
| `laya:en`, `winnow:e4b` | ollaya `/api/decide` (:11435) | Apple GPU (MLX / llama.cpp Metal) |
| `kev:4b@kevserve` | kev.serve `/v1/systemone` (:8009) | Apple GPU (MLX, bf16) |
| `decider:2b@decider` | decider-ai `uvicorn decider.serve:app` (:8010) | Apple GPU (MPS + MLX kernel) |

Companion servers (separate clones, not part of this repo):
- kev.serve: `git clone github.com/jaredpalmer/kev && uv sync --extra serve && uv run --extra serve python -m kev.serve --run jaredpalmer/kev-4b --port 8009`
- decider.serve: `pip install "decider-ai[serve,metal]" && DECIDER_MODEL=Mapika/decider-2b uvicorn decider.serve:app --host 127.0.0.1 --port 8010`

Load models only while a run needs them (no idle RAM):
```
scripts/servers.sh start    # start both GPU servers, wait for health
scripts/servers.sh stop     # stop them + unload ollaya models
scripts/servers.sh status
```

See `decision-model-benchmark-handoff-1.md` for the full research
direction, decisions already made, and open decisions.
