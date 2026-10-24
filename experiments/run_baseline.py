"""
Run baseline experiments: Standard CP vs SOTA Baselines vs FairTransCP.

This CLI provides a professional interface for running the conformal prediction
benchmark suite across various financial datasets and base classifiers.

Usage:
    python experiments/run_baseline.py run --dataset german_credit --model rf
    python experiments/run_baseline.py run --all
    python experiments/run_baseline.py run --all --sensitivity
"""

import json
import sys
from pathlib import Path

import click

# Add project root to path so we can import src/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.config import load_config
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


@click.group()
def cli():
    """FairTransCP Experiment Runner."""


@cli.command()
@click.option(
    "--dataset",
    type=click.Choice(["german_credit", "taiwan_credit", "adult_income"]),
    default=None,
    help="The dataset to use for the experiment. Defaults to config.yaml.",
)
@click.option(
    "--model",
    type=click.Choice(["rf", "xgboost", "lightgbm"]),
    default=None,
    help="The base classifier to use. Defaults to config.yaml.",
)
@click.option("--alpha", type=float, default=None, help="Target miscoverage rate.")
@click.option(
    "--score-fn", type=str, default=None, help="Nonconformity score function."
)
@click.option("--sensitive", type=str, default=None, help="Sensitive attribute name.")
@click.option(
    "--fairness-weight",
    type=float,
    default=None,
    help="Fairness trade-off weight [0, 1].",
)
@click.option(
    "--n-trials",
    type=int,
    default=None,
    help="Number of random splits to average over.",
)
@click.option(
    "--all",
    is_flag=True,
    help="Run all dataset-model combinations defined in ALL_CONFIGS.",
)
@click.option(
    "--sensitivity",
    is_flag=True,
    help="Run sensitivity analysis across alpha values [0.01, 0.05, 0.1, 0.2].",
)
def run(
    dataset,
    model,
    alpha,
    score_fn,
    sensitive,
    fairness_weight,
    n_trials,
    all,
    sensitivity,
):
    """Execute conformal prediction experiments and save results."""
    config = load_config()

    # 1. Resolve basic parameters (CLI -> Config -> Default)
    res_dataset = dataset or "german_credit"
    res_model = model or "rf"
    res_alpha = alpha if alpha is not None else config["conformal"]["alpha"]
    res_score_fn = score_fn or config["conformal"]["score_fn"]
    res_fairness_weight = (
        fairness_weight
        if fairness_weight is not None
        else config["conformal"]["fairness_weight"]
    )
    res_n_trials = (
        n_trials if n_trials is not None else config["experiment"]["n_trials"]
    )

    # 2. Resolve sensitive attribute based on dataset
    res_sensitive = sensitive or config["datasets"].get(res_dataset, {}).get(
        "sensitive", "age"
    )

    results_dir = Path(__file__).resolve().parent.parent / "results" / "tables"
    results_dir.mkdir(parents=True, exist_ok=True)

    configs = (
        ALL_CONFIGS
        if all
        else [
            {
                "dataset_name": res_dataset,
                "model_name": res_model,
                "sensitive": res_sensitive,
            }
        ]
    )

    alpha_values = [0.01, 0.05, 0.1, 0.2] if sensitivity else [res_alpha]
    all_summaries = {}

    for cfg in configs:
        tag = f"{cfg['dataset_name']}_{cfg['model_name']}"
        click.echo(f"\\n{'=' * 60}")
        click.echo(f"Running: {tag}")
        click.echo(f"{'=' * 60}")

        config_results = {}
        for a in alpha_values:
            click.echo(f"Evaluating alpha={a}...")
            results = run_experiment(
                dataset_name=cfg["dataset_name"],
                model_name=cfg["model_name"],
                alpha=a,
                score_fn=res_score_fn,
                sensitive=cfg["sensitive"],
                fairness_weight=res_fairness_weight,
                n_trials=res_n_trials,
            )
            summary = summarize_results(results)
            config_results[f"alpha_{a}"] = summary

        all_summaries[tag] = config_results

        # Save per-config results
        out_path = results_dir / f"{tag}_results.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(
                {"results": config_results, "config": cfg},
                f,
                indent=2,
            )
        click.echo(f"Saved results to {out_path}")

    # Save combined results
    if all:
        combined_path = results_dir / "all_results.json"
        with open(combined_path, "w", encoding="utf-8") as f:
            json.dump(all_summaries, f, indent=2)
        click.echo(f"All combined results saved to {combined_path}")


if __name__ == "__main__":
    cli()
