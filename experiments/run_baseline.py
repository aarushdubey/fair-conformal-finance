"""
Run baseline experiments: Standard CP vs Group-Conditional CP vs FairTransCP.

Usage:
    python experiments/run_baseline.py --dataset german_credit --model rf
    python experiments/run_baseline.py --all

This is the main experiment script for the paper. It runs all three
methods on the specified dataset and model, repeats across 10 random
splits, and saves results to results/tables/.
"""

import sys
import os
import argparse
import json
from pathlib import Path

# Add project root to path so we can import src/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.evaluation import run_experiment, summarize_results


# All dataset-model combinations we evaluate in the paper
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
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--score-fn", type=str, default="aps")
    parser.add_argument("--sensitive", type=str, default="age")
    parser.add_argument("--fairness-weight", type=float, default=0.4)
    parser.add_argument("--n-trials", type=int, default=10)
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all dataset-model combinations",
    )
    args = parser.parse_args()

    # Output directory
    results_dir = Path(__file__).resolve().parent.parent / "results" / "tables"
    results_dir.mkdir(parents=True, exist_ok=True)

    configs = ALL_CONFIGS if args.all else [
        {
            "dataset_name": args.dataset,
            "model_name": args.model,
            "sensitive": args.sensitive,
        }
    ]

    all_summaries = {}

    for cfg in configs:
        tag = f"{cfg['dataset_name']}_{cfg['model_name']}"
        print(f"\n{'='*60}")
        print(f"Running: {tag}")
        print(f"{'='*60}")

        results = run_experiment(
            dataset_name=cfg["dataset_name"],
            model_name=cfg["model_name"],
            alpha=args.alpha,
            score_fn=args.score_fn,
            sensitive=cfg["sensitive"],
            fairness_weight=args.fairness_weight,
            n_trials=args.n_trials,
        )

        summary = summarize_results(results)
        all_summaries[tag] = summary

        # Print the summary
        print(f"\nResults for {tag}:")
        for method, metrics in summary.items():
            print(f"\n  {method}:")
            for metric, value in metrics.items():
                print(f"    {metric}: {value}")

        # Save per-config results
        out_path = results_dir / f"{tag}_results.json"
        with open(out_path, "w") as f:
            json.dump(
                {"summary": summary, "config": results["config"]},
                f,
                indent=2,
            )
        print(f"\nSaved to {out_path}")

    # Save combined results
    if args.all:
        combined_path = results_dir / "all_results.json"
        with open(combined_path, "w") as f:
            json.dump(all_summaries, f, indent=2)
        print(f"\nAll results saved to {combined_path}")


if __name__ == "__main__":
    main()
