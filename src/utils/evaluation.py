"""
Evaluation pipeline for running experiments end-to-end.

This ties together the data loading, model training, conformal
calibration, and fairness evaluation into a single function call.
The idea is that experiment scripts should be short and readable,
with all the plumbing hidden here.
"""

import numpy as np
from sklearn.model_selection import train_test_split
from typing import Dict, Any, List

from ..data.loaders import load_dataset
from ..models.classifiers import get_classifier
from ..conformal.base import SplitConformalClassifier
from ..conformal.fair_conformal import FairTransCP
from ..fairness.metrics import compute_all_metrics


def run_experiment(
    dataset_name: str,
    model_name: str,
    alpha: float = 0.1,
    score_fn: str = "aps",
    sensitive: str = "gender",
    fairness_weight: float = 0.4,
    test_size: float = 0.2,
    cal_fraction: float = 0.3,
    random_state: int = 42,
    n_trials: int = 10,
) -> Dict[str, Any]:
    """
    Run a complete experiment and return all metrics.

    The data is split three ways:
        train (for fitting the base model)
        calibration (for conformal calibration)
        test (for evaluation)

    We run multiple trials with different random splits to get
    error bars (important for small datasets like German Credit).

    Parameters
    ----------
    dataset_name : str
        Which dataset to use.
    model_name : str
        Which base classifier ('rf', 'xgboost', 'lightgbm').
    alpha : float
        Miscoverage rate.
    score_fn : str
        Nonconformity score function.
    sensitive : str
        Which sensitive attribute to use.
    fairness_weight : float
        Trade-off parameter for FairTransCP.
    test_size : float
        Fraction held out for testing.
    cal_fraction : float
        Fraction of the remaining data used for calibration
        (the rest is for training the base model).
    random_state : int
        Base random seed. Each trial uses random_state + trial_idx.
    n_trials : int
        How many random splits to average over.

    Returns
    -------
    dict with keys:
        'baseline_metrics': list of metric dicts (one per trial)
        'group_cp_metrics': list of metric dicts
        'fairtranscp_metrics': list of metric dicts
        'config': experiment configuration
    """
    # Load the dataset once
    data = load_dataset(dataset_name, sensitive=sensitive)
    X, y, sens = data["X"], data["y"], data["sensitive"]

    baseline_results = []
    group_cp_results = []
    fair_results = []

    for trial in range(n_trials):
        seed = random_state + trial

        # Three-way split: train / calibration / test
        X_trainval, X_test, y_trainval, y_test, s_trainval, s_test = (
            train_test_split(
                X, y, sens, test_size=test_size, random_state=seed, stratify=y
            )
        )
        X_train, X_cal, y_train, y_cal, s_train, s_cal = train_test_split(
            X_trainval,
            y_trainval,
            s_trainval,
            test_size=cal_fraction,
            random_state=seed,
            stratify=y_trainval,
        )

        # Fit the base model on training data only
        clf = get_classifier(model_name, random_state=seed)
        clf.fit(X_train, y_train)

        # Get probability estimates
        prob_cal = clf.predict_proba(X_cal)
        prob_test = clf.predict_proba(X_test)

        # ── Method 1: Standard (marginal) conformal prediction ──
        standard_cp = SplitConformalClassifier(
            alpha=alpha, score_fn=score_fn, random_state=seed
        )
        standard_cp.calibrate(prob_cal, y_cal)
        sets_baseline = standard_cp.predict_sets(prob_test)
        baseline_results.append(
            compute_all_metrics(sets_baseline, y_test, s_test)
        )

        # ── Method 2: Group-conditional CP (equalized coverage) ──
        # This is the Romano et al. baseline where we calibrate
        # separately per group. Pure coverage equalization.
        group_cp = FairTransCP(
            alpha=alpha,
            score_fn=score_fn,
            fairness_weight=0.0,  # Pure coverage equalization
            random_state=seed,
        )
        group_cp.calibrate(prob_cal, y_cal, s_cal)
        sets_group = group_cp.predict_sets(prob_test, s_test)
        group_cp_results.append(
            compute_all_metrics(sets_group, y_test, s_test)
        )

        # ── Method 3: FairTransCP (our method) ──
        fair_cp = FairTransCP(
            alpha=alpha,
            score_fn=score_fn,
            fairness_weight=fairness_weight,
            random_state=seed,
        )
        fair_cp.calibrate(prob_cal, y_cal, s_cal)
        sets_fair = fair_cp.predict_sets(prob_test, s_test)
        fair_results.append(
            compute_all_metrics(sets_fair, y_test, s_test)
        )

    return {
        "baseline_metrics": baseline_results,
        "group_cp_metrics": group_cp_results,
        "fairtranscp_metrics": fair_results,
        "config": {
            "dataset": dataset_name,
            "model": model_name,
            "alpha": alpha,
            "score_fn": score_fn,
            "sensitive": sensitive,
            "fairness_weight": fairness_weight,
            "n_trials": n_trials,
            "test_size": test_size,
            "cal_fraction": cal_fraction,
        },
    }


def summarize_results(results: Dict[str, Any]) -> Dict[str, Dict[str, str]]:
    """
    Aggregate trial results into mean +/- std format.

    Returns a dict of {method_name: {metric: "mean +/- std"}} that's
    ready to be printed as a table in the paper.
    """
    summary = {}

    for method_key, method_name in [
        ("baseline_metrics", "Standard CP"),
        ("group_cp_metrics", "Group-Conditional CP"),
        ("fairtranscp_metrics", "FairTransCP (Ours)"),
    ]:
        trial_metrics = results[method_key]
        # Collect all metric keys from the first trial
        metric_keys = [
            k
            for k in trial_metrics[0].keys()
            if not k.startswith("coverage_") and not k.startswith("avg_size_")
        ]

        method_summary = {}
        for key in metric_keys:
            values = [t[key] for t in trial_metrics if key in t]
            mean = np.mean(values)
            std = np.std(values)
            method_summary[key] = f"{mean:.4f} +/- {std:.4f}"

        summary[method_name] = method_summary

    return summary
