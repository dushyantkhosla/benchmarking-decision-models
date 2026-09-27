"""CLI entry point for the decision-model benchmark.

Examples:
    python run_benchmark.py --dataset banking77 --models laya:en,winnow:e4b --tiers reduced,full
    python run_benchmark.py --dataset jigsaw --models laya:en,winnow:e4b
"""

from __future__ import annotations

import argparse
from pathlib import Path

from core.runner import run
from datasets import banking77, jigsaw

REPO_ROOT = Path(__file__).parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="banking77", choices=["banking77", "jigsaw"])
    parser.add_argument("--models", required=True, help="comma-separated model specs")
    parser.add_argument("--tiers", default="reduced,full", help="banking77 difficulty tiers")
    parser.add_argument("--n", type=int, default=500, help="banking77 sample size")
    parser.add_argument("--n-pos", type=int, default=42, help="jigsaw positives per criterion")
    parser.add_argument("--n-neg", type=int, default=42, help="jigsaw negatives per criterion")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--distractors", type=int, default=9, help="banking77 distractors")
    parser.add_argument("--split", default="test")
    parser.add_argument("--out", default="results")
    parser.add_argument("--host", default="http://localhost:11435", help="ollaya host")
    parser.add_argument("--kev-host", default="http://localhost:8009", help="kev.serve host (models tagged @kevserve)")
    parser.add_argument("--decider-host", default="http://localhost:8010", help="decider-ai server host (models tagged @decider)")
    parser.add_argument("--resume", default=None, help="existing run dir to resume/extend")
    args = parser.parse_args()

    if args.dataset == "banking77":
        csv_path, sha256 = banking77.download(REPO_ROOT / "data" / "banking77", split=args.split)
        rows = banking77.load_rows(csv_path)
        labels = banking77.load_labels(rows)
        print(f"loaded {len(rows)} rows, {len(labels)} labels (sha256 {sha256[:12]}…)")

        sample = banking77.stratified_sample(rows, n=args.n, seed=args.seed)
        tiers = [t.strip() for t in args.tiers.split(",")]
        examples_by_tier = {
            tier: banking77.build_examples(
                sample, labels, tier=tier, seed=args.seed, split=args.split,
                n_distractors=args.distractors,
            )
            for tier in tiers
        }
        dataset_meta = {
            "name": "banking77",
            "source_url": banking77.TEST_URL if args.split == "test" else banking77.TRAIN_URL,
            "sha256": sha256,
            "split": args.split,
            "rows": len(rows),
            "labels": len(labels),
        }
        config_meta = {
            "n": args.n,
            "seed": args.seed,
            "distractors": args.distractors,
            "tiers": tiers,
            "instructions": banking77.INSTRUCTIONS,
        }
    else:  # jigsaw
        files = jigsaw.download(REPO_ROOT / "data" / "jigsaw")
        rows = jigsaw.load_rows(files["test"][0], files["test_labels"][0])
        print(f"loaded {len(rows)} scored rows (test sha256 {files['test'][1][:12]}…)")

        pairs = jigsaw.sample_pairs(rows, n_pos=args.n_pos, n_neg=args.n_neg, seed=args.seed)
        examples_by_tier = {"binary": jigsaw.build_examples(pairs, split=args.split)}
        dataset_meta = {
            "name": "jigsaw",
            "source_url": jigsaw.BASE_URL,
            "sha256": {name: sha for name, (_, sha) in files.items()},
            "split": args.split,
            "rows_scored": len(rows),
            "criteria": jigsaw.CRITERIA,
        }
        config_meta = {
            "n_pos": args.n_pos,
            "n_neg": args.n_neg,
            "seed": args.seed,
            "questions": jigsaw.QUESTIONS,
        }

    out_dir = run(
        models=[m.strip() for m in args.models.split(",")],
        examples_by_tier=examples_by_tier,
        out_root=REPO_ROOT / args.out,
        hosts={
            "ollaya": args.host,
            "kevserve": args.kev_host,
            "decider": args.decider_host,
        },
        dataset_meta=dataset_meta,
        config_meta=config_meta,
        repo_root=REPO_ROOT,
        resume_dir=(REPO_ROOT / args.resume) if args.resume else None,
    )
    print(f"artifacts written to {out_dir}")


if __name__ == "__main__":
    main()
