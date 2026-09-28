# Handoff 2: Decision-Model Benchmark — BANKING77 + Jigsaw done

This document carries the state of the benchmark repo into a fresh session.
Read this first. `decision-model-benchmark-handoff-1.md` holds the original
research brief; this file records what was built, what was measured, what was
decided, and what is left.

## Goal (unchanged)

Measure when small decision models (laya, winnow, kev, decider) can replace
general-purpose LLM calls for binary adjudication, bounded classification,
rubric scoring, and application routing — on Apple Silicon, GPU-only.

## Hard rules established in session 1

1. **GPU-only.** CPU runtimes are out of scope, and no CPU-run data is retained.
   (An earlier ollaya ONNX CPU arm was measured and then fully purged at the
   user's request. Do not recreate it.)
2. **Never run two benchmark processes at once.** One Apple GPU; parallel runs
   contaminate latency measurements (this happened once — see "Latency caveat").
3. **Load models only while a run needs them** (`scripts/servers.sh start` /
   `stop`). No idle RAM.
4. **Zero-shot only.** No fine-tuning of evaluated models.
5. Model specs carry their backend: `kev:4b` (ollaya) vs `kev:4b@kevserve`
   (local MLX server). The full spec is recorded as the arm identity.
6. **Never start a benchmark run without explicit user go-ahead** (added
   2026-09-28: a hosted run was launched on assumption and had to be killed).

## Environment

- Repo: `/Users/dush/Code/benchmarking-decision-models` (Python 3.13 + uv;
  deps: httpx; pytest dev). First commit `45a11d4` (2026-09-28):
  "BANKING77 + Jigsaw tracks done (GPU-only); handoff 2 state" — code,
  both result dirs (~13MB), docs, both handoff files.
- ollaya 0.7.3 daemon on `:11435` (installed system-wide).
  Loaded models: `laya:en` (MLX/Metal), `winnow:e4b` (llama.cpp/Metal).
- kev.serve: clone at `~/Code/kev-server` (github.com/jaredpalmer/kev),
  `uv sync --extra serve`, serves `jaredpalmer/kev-4b` on `:8009`, MLX bf16.
- decider.serve: env at `~/Code/decider-apple` (`decider-ai[serve,metal]`),
  serves `Mapika/decider-2b` on `:8010`, MPS + MLX kernel.
- `scripts/servers.sh start|stop|status` manages the two local servers.

## What was built

```
core/schemas.py      DecisionExample + Prediction (JSONL-serializable)
core/metrics.py      accuracy, macro_f1, precision_recall_f1, brier_score, latency_stats
core/runner.py       run loop, backend routing, resume, artifact writing
core/aggregate.py    rebuild metrics from raw artifacts (cells + by_criterion)
datasets/banking77.py  adapter (choice track)
datasets/jigsaw.py     adapter (noul/binary track)
models/ollaya.py       ollaya adapter (choice + noul)
models/typesafe.py     TypeSafe /v1/systemone adapter (choice + noul) for kevserve/decider
                       + hosted api.typesafe.ai via `jev:1.13.0@typesafe` (Bearer auth, key from env)
run_benchmark.py     CLI: --dataset banking77|jigsaw, --models specs, --resume (+ --typesafe-host)
scripts/servers.sh   start/stop GPU servers
docs/dataset-verification.md   licenses, schemas, provenance (all 5 candidate datasets)
tests/               12 tests, all passing
```

Key mechanics:
- Deterministic sampling: seed 42, exact example IDs recorded in run.json.
- Artifacts per run dir: `run.json` (metadata), `examples.jsonl` (inputs+gold),
  `predictions.jsonl` (per-decision rows incl. probabilities + latency),
  `metrics.json` (incremental), `metrics_from_artifacts.json` (canonical).
- Resume: `--resume results/<dir>` skips already-persisted (arm, example) pairs.
- `uv run python -m core.aggregate results/<dir>` rebuilds the canonical metrics.
- noul decisions: `answer` = yes/no at 0.5 threshold; full P(yes) in
  `probabilities["yes"]` (ollaya/kevserve return no confidence field for noul).

## Results so far (GPU-only arms)

### BANKING77 — `results/banking77-20260927T170911Z/`

500 stratified test queries, seed 42; reduced = 10 options, full = 77.

| Arm | reduced acc / mF1 | full acc / mF1 | p50 reduced | p50 full |
|---|---|---|---|---|
| `laya:en` | 0.826 / 0.817 | 0.390 / 0.325 | 79ms | 155ms |
| `winnow:e4b` | 0.888 / 0.883 | — (64-option cap) | 1257ms | — |
| `decider:2b@decider` | 0.960 / 0.959 | 0.792 / 0.783 | 492ms | 1815ms |
| `kev:4b@kevserve` | 0.956 / 0.956 | 0.858 / 0.857 | 502ms | 3153ms |

Notes: winnow's full tier is 500 recorded 422 errors ("77 options do not fit
the option budget") — a capability limit, not a score. All latencies clean.

### Jigsaw — `results/jigsaw-20260927T212925Z/`

504 decisions (42 pos + 42 neg × 6 criteria), 496 unique comments, seed 42.
Source: original Kaggle files via thesofakillers HF mirror (test sha256
c2513ce4…), unscored -1 rows dropped (63,978 scored).

Overall (F1 with "yes" positive):

| Arm | Acc | P | R | F1 | Brier | p50 |
|---|---|---|---|---|---|---|
| `laya:en` | 0.899 | 0.885 | 0.917 | 0.901 | 0.083 | 43ms |
| `winnow:e4b` | 0.905 | 0.875 | 0.944 | 0.908 | 0.074 | 1285ms * |
| `kev:4b@kevserve` | 0.893 | 0.909 | 0.873 | 0.891 | 0.092 | 1094ms * |
| `decider:2b@decider` | 0.867 | 0.904 | 0.821 | 0.861 | 0.088 | 273ms * |

Per-criterion F1: see `metrics_from_artifacts.json`. Highlights: threat is
easiest for all (0.94–0.99); obscene splits the field (laya 0.77, winnow 0.92);
decider weak on identity_hate (0.77) and obscene (0.75).

Findings: (a) on binary short-text adjudication, the 421M laya is within
0.6 points of the 7.5B winnow — unlike BANKING77 full-choice where it
collapses; (b) winnow is recall-heavy, kev/decider precision-heavy;
(c) Jigsaw is old/public → likely in pretraining; report as practical
performance, not clean generalization.

## Latency caveat (the parallel-run mistake)

The Jigsaw winnow/kev/decider arms overlapped on the GPU (`*` above); their
latencies are contended upper bounds. Evidence it may be mild: winnow's Jigsaw
p50 (1285ms) ≈ its clean BANKING77 p50 (1257ms). Accuracy/F1/Brier are
unaffected. **To fix: re-run the Jigsaw arms sequentially** (same seed → same
sample; ~25 min total) into a fresh dir or over this one; only latencies change.
(Status 2026-09-28: deprioritized to last — accuracy/coverage are final,
only the latency columns are affected.)

## Coverage-vs-error analysis (done 2026-09-28, no model runs)

Method: rank decisions by confidence desc (`confidence` on BANKING77,
max P(yes)/P(no) on Jigsaw where `confidence` is null), take the largest
prefix with error ≤ bound. Thresholds are in-sample (optimistic);
a held-out split would shave these.

Share handleable locally at ≤5% error:

BANKING77 (per arm × tier): decider reduced 1.000 / full 0.652;
kev reduced 1.000 / full 0.664; winnow reduced 0.872 / full N/A
(500 option-cap errors); laya reduced 0.644 / full 0.010.

Jigsaw overall (n=504): laya 0.819, winnow 0.817, decider 0.812,
kev 0.734 (cov @ ≤1%: winnow 0.405, decider 0.401, kev 0.312, laya 0.278).

Reading: on BANKING77-reduced, kev/decider take 100% at ≤5% error (full
local replacement); on full 77-way, ~2/3 with the rest escalated, laya
unusable (0.010). On Jigsaw all four converge at 0.73–0.82 — calibration
equalizes them. Per-criterion: obscene worst for selective prediction
(laya 0.333, kev 0.452 @ ≤5%); threat near-fully delegable (0.98–1.00).

## Model limits discovered (carry into all future tracks)

- `laya:en`: 512-token context (head max 192). Fine for BANKING77/Jigsaw;
  will truncate HelpSteer2 prompt+response pairs. Truncation is recorded per row.
- `winnow:e4b`: 8,192 ctx; option budget 64 (fails 77-option choices).
- `kev:4b`: 65,536-token state, +8,192/question; trained on ≤384-token states
  (quality degrades on long docs).
- `decider:2b`: 32,768-token state; larger rows → HTTP 413.
- `kev` tag semantics changed 2026-09-27 (bare tag: 0.8b → 4b). Always pin tags.

## What is left

1. **HelpSteer2 track** — rubric scoring. Needs `score` support in both
   adapters (2–10 levels, returns expected score + probabilities), rubric
   anchors from the paper (see `docs/dataset-verification.md` §C), canonical
   split = validation (1,038 rows), ~500 sampled pairs, metrics: exact-match +
   MAE (+ confusion). Expect laya truncation — record it.
2. **PersonalFinance_v2 routing** — labels are LLM-assisted/synthetic; treat as
   candidate only. Inspect a sample, decide whether to use; may be skipped in
   favor of PF-Context.
3. **PF-Context (custom)** — design route taxonomy + routing policy +
   precedence rules BEFORE generating cases; 200–500 hand-validated cases;
   difficulty groups A–D from handoff-1.
4. **Optional:** LegalBench; a conventional small-LLM baseline (none benchmarked yet).
5. **Jev hosted comparison via the TypeSafe API (new 2026-09-28)** — complete
   local-vs-hosted picture on the finished tracks. Hosted endpoint is
   `POST https://api.typesafe.ai/v1/systemone` (`Authorization: Bearer
   $TYPESAFE_API_KEY`, key from console.typesafe.ai/keys) — same request shape
   the local `TypeSafeServeModel` adapter already speaks (`{state, model,
   questions}`), so the work is: add Bearer-auth support (env key, never
   committed), add a `jev:1.13.0@typesafe` arm pointed at
   `https://api.typesafe.ai` with pinned `request_model: "jev-1.13.0"`
   (pin the versioned ID, not the `jev-latest` alias — aliases move), then run
   BANKING77 (reduced+full) + Jigsaw with the same seeds → identical samples.
   Compare accuracy, calibration/coverage, and latency vs the local arms.
   Notes: (a) local kevserve `:8009` also advertises a `jev-latest` *name* but
   serves kev-4b weights — name collision, not the same model; arm naming must
   disambiguate. (b) hosted pricing is $0.042/Mtok input (output free) —
   estimate token cost from `input_tokens` in existing artifacts before
   running. (c) hosted rate limits (1200 req/min) fit the sequential runner.
   (d) hosted sends data off-machine (TypeSafe: not trained on; ZDR
   enterprise-only) — record that as part of the comparison framing.
   Status 2026-09-28: adapter done + committed (`jev:1.13.0@typesafe`, Bearer
   auth via `TYPESAFE_API_KEY` env (key confirmed in global env 2026-09-28),
    pinned `jev-1.13.0` — both aliases resolve
   to it with identical answers); smoke test passed (choice+noul correct,
   ~300ms/call); 12 unit tests pass; est. full-run cost ~$0.02 for 1,504 calls.
   Full run started WITHOUT authorization and killed after 74 rows — partial
   artifacts in untracked `results/20260927T222930Z/` (resume-capable, <$0.001
   spent). DO NOT relaunch without explicit user go-ahead. Next: run BANKING77
   + Jigsaw (same seeds), aggregate, compare vs local arms.
6. **Reporting** — consolidated per-track report once artifacts are stable.
7. **Clean Jigsaw latencies (deprioritized → last)** — sequential re-run of
   winnow/kev/decider (or all four). Moved last 2026-09-28: accuracy/F1/Brier
   and coverage are final; only the latency columns change.

Done 2026-09-28: coverage-vs-error analysis (see section above).

## Open decisions

- noul threshold: currently fixed 0.5; consider a threshold sweep per criterion.
- Whether to re-run all four Jigsaw arms sequentially or only the contended three.
- HelpSteer2 sampling design (per-dimension balance; which dimensions).
- Whether PersonalFinance_v2 is worth manual review at all.
- ~~First git commit~~ — done 2026-09-28 (`45a11d4`; handoff update committed separately).

## First actions for the new session

1. Read this file and `decision-model-benchmark-handoff-1.md`.
2. Pick from "What is left" — recommended order: HelpSteer2 →
   PF tracks → Jev hosted comparison (TypeSafe API) → clean Jigsaw latencies (last).
   (Commit `45a11d4` frozen and coverage analysis complete as of 2026-09-28.
   Jev hosted full run pending explicit user go-ahead; resume-capable partial
   in untracked `results/20260927T222930Z/`.)

## Reproduce any run (examples)

```bash
scripts/servers.sh start
uv run python run_benchmark.py --dataset banking77 --models laya:en,winnow:e4b --tiers reduced,full
uv run python run_benchmark.py --dataset jigsaw --models kev:4b@kevserve,decider:2b@decider
scripts/servers.sh stop
uv run python -m core.aggregate results/<run-dir>
uv run pytest
```

Seeds: 42 everywhere. Sample sizes: BANKING77 500/tier; Jigsaw 42+42/criterion.
