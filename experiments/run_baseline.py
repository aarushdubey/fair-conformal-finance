"""
Run baseline experiments: Standard CP vs SOTA Baselines vs FairTransCP.

Usage:
    python experiments/run_baseline.py --dataset german_credit --model rf
    python experiments/run_baseline.py --all
    python experiments/run_baseline.py --all --sensitivity

Runs all methods on the specified dataset and model, repeats across trials,
and saves results to results/tables/.
"""

import sys
import os
import argparse
import json
from pathlib import Path

# Add project root to path so we can import src/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.evaluation import run_experiment, summarize_results


# All dataset-model combinations evaluated in the paper (strictly authentic real benchmarks)
ALL_CONFIGS = [
    {"dataset_name": "german_credit", "model_name": "rf", "sensitive": "age"},
    {"dataset_name": "german_credit", "model_name": "xgboost", "sensitive": "age"},
    {"dataset_name": "german_credit", "model_name": "lightgbm", "sensitive": "age"},
    {"dataset_name": "taiwan_credit", "model_name": "rf", "sensitive": "gender"},
    {"dataset_name": "taiwan_credit", "model_name": "xgboost", "sensitive": "gender"},
    {"dataset_name": "taiwan_credit", "model_name": "lightgbm", "sensitive": "gender"},
    {"dataset_name": "adult_income", "model_name": "rf", "sensitive": "gender"},
    {"dataset_name": "adult_income", "model_name": "xgboost", "sensitive": "gender"},
    {"dataset_name": "adult_income", "model_name": "lightgbm", "sensitive": "gender"},
]


def main():
    parser = argparse.ArgumentParser(
        description="Run conformal prediction experiments"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="german_credit",
        choices=["german_credit", "taiwan_credit", "adult_income"],
    )
    parser.add_argument(
        "--model",
        type=str,
        default="rf",
        choices=["rf", "xgboost", "lightgbm"],
    )
    parser.add_argument("--alpha", type=float, default=0.1, help="Target miscoverage rate")
    parser.add_argument("--score-fn", type=str, default="aps")
    parser.add_argument("--sensitive", type=str, default="age")
    parser.add_argument("--fairness-weight", type=float, default=0.4)
    parser.add_argument("--n-trials", type=int, default=5)
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all dataset-model combinations",
    )
    parser.add_argument(
        "--sensitivity",
        action="store_true",
        help="Run sensitivity analysis across alpha in [0.01, 0.05, 0.1, 0.2]",
    )
    args = parser.parse_args()

    results_dir = Path(__file__).resolve().parent.parent / "results" / "tables"
    results_dir.mkdir(parents=True, exist_ok=True)

    configs = ALL_CONFIGS if args.all else [
        {
            "dataset_name": args.dataset,
            "model_name": args.model,
            "sensitive": args.sensitive,
        }
    ]

    alpha_values = [0.01, 0.05, 0.1, 0.2] if args.sensitivity else [args.alpha]
    all_summaries = {}

    for cfg in configs:
        tag = f"{cfg['dataset_name']}_{cfg['model_name']}"
        print(f"\n{'='*60}")
        print(f"Running: {tag}")
        print(f"{'='*60}")

        config_results = {}
        for alpha in alpha_values:
            print(f"Evaluating alpha={alpha}...")
            results = run_experiment(
                dataset_name=cfg["dataset_name"],
                model_name=cfg["model_name"],
                alpha=alpha,
                score_fn=args.score_fn,
                sensitive=cfg["sensitive"],
                fairness_weight=args.fairness_weight,
                n_trials=args.n_trials,
            )
            summary = summarize_results(results)
            config_results[f"alpha_{alpha}"] = summary

        all_summaries[tag] = config_results

        # Save per-config results
        out_path = results_dir / f"{tag}_results.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(
                {"results": config_results, "config": cfg},
                f,
                indent=2,
            )
        print(f"Saved results to {out_path}")

    # Save combined results
    if args.all:
        combined_path = results_dir / "all_results.json"
        with open(combined_path, "w", encoding="utf-8") as f:
            json.dump(all_summaries, f, indent=2)
        print(f"All combined results saved to {combined_path}")


if __name__ == "__main__":
    main()
