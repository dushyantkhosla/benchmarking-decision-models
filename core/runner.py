"""Benchmark runner: dataset examples -> model adapter -> persisted artifacts.

Artifacts per run, under results/<run_id>/:
    run.json          metadata + config (reproducibility record)
    predictions.jsonl one normalized Prediction per line (appended incrementally)
    metrics.json      aggregate metrics per (model, tier)
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from core.metrics import accuracy, latency_stats, macro_f1
from core.schemas import DecisionExample, Prediction
from models.ollaya import OllayaConfig, OllayaDecisionModel
from models.typesafe import TypeSafeConfig, TypeSafeServeModel


def _make_adapter(spec: str, hosts: dict[str, str]):
    """Model spec -> adapter. A `@backend` suffix selects the backend (default: ollaya).

    Examples: `kev:4b`, `winnow:e4b` (ollaya);
              `kev:4b@kevserve` (jaredpalmer/kev's local MLX server);
              `decider:2b@decider` (decider-ai's local server).
    """
    # The full spec (suffix included) is the adapter's model name, so predictions,
    # resume keys, and metric keys all identify the exact runtime arm.
    if spec.endswith("@kevserve"):
        return TypeSafeServeModel(
            spec, TypeSafeConfig(host=hosts["kevserve"], request_model="kev-latest")
        )
    if spec.endswith("@decider"):
        return TypeSafeServeModel(
            spec,
            TypeSafeConfig(host=hosts["decider"], request_model=spec.removesuffix("@decider")),
        )
    return OllayaDecisionModel(spec, OllayaConfig(host=hosts["ollaya"]))


def _describe_model(spec: str, hosts: dict[str, str]) -> dict:
    try:
        adapter = _make_adapter(spec, hosts)
        if isinstance(adapter, OllayaDecisionModel):
            return adapter.show()
        return {"backend": type(adapter).__name__, "host": adapter.config.host, "models": adapter.models()}
    except Exception as exc:
        return {"error": str(exc)}


def _git_metadata(repo_root: Path) -> dict:
    def git(*args: str) -> str | None:
        try:
            return subprocess.run(
                ["git", *args], cwd=repo_root, capture_output=True, text=True, check=True
            ).stdout.strip()
        except Exception:
            return None

    return {
        "commit": git("rev-parse", "HEAD"),
        "dirty": bool(git("status", "--porcelain")),
    }


def _hardware_metadata() -> dict:
    meta = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": sys.version.split()[0],
    }
    for key, cmd in {
        "chip": ["sysctl", "-n", "machdep.cpu.brand_string"],
        "memory_bytes": ["sysctl", "-n", "hw.memsize"],
    }.items():
        try:
            meta[key] = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout.strip()
        except Exception:
            pass
    return meta


def run(
    models: list[str],
    examples_by_tier: dict[str, list[DecisionExample]],
    out_root: Path,
    hosts: dict[str, str],
    dataset_meta: dict,
    config_meta: dict,
    repo_root: Path,
    resume_dir: Path | None = None,
) -> Path:
    if resume_dir is not None:
        out_dir = resume_dir
        out_dir.mkdir(parents=True, exist_ok=True)
    else:
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        out_dir = out_root / run_id
        out_dir.mkdir(parents=True, exist_ok=False)
    predictions_path = out_dir / "predictions.jsonl"

    # Resume: skip (model, example_id) pairs already persisted.
    completed: set[tuple[str, str]] = set()
    if predictions_path.exists():
        for line in predictions_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                completed.add((rec["model"], rec["example_id"]))

    run_doc = {
        "run_id": out_dir.name,
        "timestamp_utc": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "resumed": resume_dir is not None,
        "git": _git_metadata(repo_root),
        "hardware": _hardware_metadata(),
        "hosts": hosts,
        "services": {},
        "dataset": dataset_meta,
        "config": config_meta,
        "example_ids": {tier: [e.id for e in exs] for tier, exs in examples_by_tier.items()},
        "models": {},
    }

    # Probe servers + record per-model artifact details before benchmarking.
    plain = [m for m in models if "@" not in m]
    if plain:
        try:
            run_doc["services"]["ollaya_version"] = OllayaDecisionModel(plain[0]).version()
        except Exception as exc:
            run_doc["services"]["ollaya_version_error"] = str(exc)
    for model in models:
        run_doc["models"][model] = _describe_model(model, hosts)

    # Write metadata + the full example set (with gold) BEFORE benchmarking,
    # so a crash still leaves a reproducible, self-contained artifact.
    (out_dir / "run.json").write_text(json.dumps(run_doc, indent=2, ensure_ascii=False))
    (out_dir / "examples.jsonl").write_text(
        "".join(
            json.dumps(e.to_dict(), ensure_ascii=False) + "\n"
            for exs in examples_by_tier.values()
            for e in exs
        ),
        encoding="utf-8",
    )

    metrics_path = out_dir / "metrics.json"
    aggregates: dict[str, dict] = {}
    if metrics_path.exists():
        aggregates = json.loads(metrics_path.read_text(encoding="utf-8"))

    with predictions_path.open("a", encoding="utf-8") as out:
        for model in models:
            adapter = _make_adapter(model, hosts)
            adapter.warmup()
            for tier, examples in examples_by_tier.items():
                key = f"{model} | {tier}"
                todo = [e for e in examples if (model, e.id) not in completed]
                if not todo:
                    print(f"{key}: already complete, skipping", flush=True)
                    continue
                preds: list[Prediction] = []
                t_start = time.perf_counter()
                for example in todo:
                    pred = adapter.decide(example)
                    preds.append(pred)
                    out.write(json.dumps(pred.to_dict(), ensure_ascii=False) + "\n")
                    out.flush()
                elapsed = time.perf_counter() - t_start

                ok = [p for p in preds if p.error is None]
                golds = [e.gold for e, p in zip(todo, preds) if p.error is None]
                answers = [p.answer for p in ok]
                latencies = [p.latency_ms for p in preds if p.latency_ms is not None]
                aggregates[key] = {
                    "n": len(preds),
                    "errors": len(preds) - len(ok),
                    "truncated": sum(1 for p in preds if p.state_truncated),
                    "accuracy": accuracy(golds, answers),
                    "macro_f1": macro_f1(golds, answers),
                    "latency_ms": latency_stats(latencies),
                    "decisions_per_second": len(preds) / elapsed if elapsed else 0.0,
                    "mean_input_tokens": (
                        sum(p.input_tokens for p in ok if p.input_tokens) / len(ok) if ok else 0.0
                    ),
                }
                # Persist metrics after every cell, not just at the end.
                metrics_path.write_text(json.dumps(aggregates, indent=2, ensure_ascii=False))
                print(
                    f"{key}: acc={aggregates[key]['accuracy']:.3f} "
                    f"macroF1={aggregates[key]['macro_f1']:.3f} "
                    f"p50={aggregates[key]['latency_ms']['p50']:.0f}ms "
                    f"errors={aggregates[key]['errors']} n={aggregates[key]['n']}",
                    flush=True,
                )

    return out_dir
