"""
Evaluation pipeline for running experiments end-to-end.

Ties together data loading, model training, conformal calibration
across multiple SOTA baselines and FairTransCP, and fairness evaluation.
"""

import numpy as np
from sklearn.model_selection import train_test_split
from typing import Dict, Any, List

from ..data.loaders import load_dataset
from ..models.classifiers import get_classifier
from ..conformal.base import SplitConformalClassifier
from ..conformal.fair_conformal import FairTransCP
from ..conformal.baselines import MondrianCP, LCCP, GenericFairCP
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
    optimize_lambda: bool = True,
) -> Dict[str, Any]:
    """
    Run a complete experiment across all conformal methods and return metrics.

    The data is split three ways:
        train (for fitting the base model)
        calibration (for conformal calibration)
        test (for evaluation)
    """
    # Load dataset
    data = load_dataset(dataset_name, sensitive=sensitive)
    X, y, sens = data["X"], data["y"], data["sensitive"]

    baseline_results = []
    mondrian_results = []
    lccp_results = []
    generic_fair_results = []
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

        # Fit base model
        clf = get_classifier(model_name, random_state=seed)
        clf.fit(X_train, y_train)

        # Predicted probabilities
        prob_cal = clf.predict_proba(X_cal)
        prob_test = clf.predict_proba(X_test)

        # 1. Standard (marginal) CP
        standard_cp = SplitConformalClassifier(
            alpha=alpha, score_fn=score_fn, random_state=seed
        )
        standard_cp.calibrate(prob_cal, y_cal)
        sets_baseline = standard_cp.predict_sets(prob_test)
        baseline_results.append(
            compute_all_metrics(sets_baseline, y_test, s_test)
        )

        # 2. Mondrian CP
        mondrian_cp = MondrianCP(
            alpha=alpha, score_fn=score_fn, random_state=seed
        )
        mondrian_cp.calibrate(prob_cal, y_cal, s_cal)
        sets_mondrian = mondrian_cp.predict_sets(prob_test, s_test)
        mondrian_results.append(
            compute_all_metrics(sets_mondrian, y_test, s_test)
        )

        # 3. LC-CP (Label-Clustered CP)
        lccp = LCCP(
            alpha=alpha, score_fn=score_fn, random_state=seed
        )
        lccp.calibrate(prob_cal, y_cal, s_cal)
        sets_lccp = lccp.predict_sets(prob_test, s_test)
        lccp_results.append(
            compute_all_metrics(sets_lccp, y_test, s_test)
        )

        # 4. Generic Fair CP
        generic_fair = GenericFairCP(
            alpha=alpha, score_fn=score_fn, random_state=seed
        )
        generic_fair.calibrate(prob_cal, y_cal, s_cal)
        sets_generic = generic_fair.predict_sets(prob_test, s_test)
        generic_fair_results.append(
            compute_all_metrics(sets_generic, y_test, s_test)
        )

        # 5. Group-conditional CP
        group_cp = FairTransCP(
            alpha=alpha,
            score_fn=score_fn,
            fairness_weight=0.0,
            random_state=seed,
        )
        group_cp.calibrate(prob_cal, y_cal, s_cal)
        sets_group = group_cp.predict_sets(prob_test, s_test)
        group_cp_results.append(
            compute_all_metrics(sets_group, y_test, s_test)
        )

        # 6. FairTransCP (Ours)
        fair_cp = FairTransCP(
            alpha=alpha,
            score_fn=score_fn,
            fairness_weight=fairness_weight,
            random_state=seed,
            optimize_lambda=optimize_lambda,
        )
        fair_cp.calibrate(prob_cal, y_cal, s_cal)
        sets_fair = fair_cp.predict_sets(prob_test, s_test)
        fair_results.append(
            compute_all_metrics(sets_fair, y_test, s_test)
        )

    return {
        "baseline_metrics": baseline_results,
        "mondrian_metrics": mondrian_results,
        "lccp_metrics": lccp_results,
        "generic_fair_metrics": generic_fair_results,
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
    Summarize multiple trials into mean +/- std for key metrics.
    """
    summary = {}
    methods = {
        "Standard CP": results["baseline_metrics"],
        "Mondrian CP": results.get("mondrian_metrics", []),
        "LC-CP": results.get("lccp_metrics", []),
        "Generic Fair CP": results.get("generic_fair_metrics", []),
        "Group-Conditional CP": results["group_cp_metrics"],
        "FairTransCP (Ours)": results["fairtranscp_metrics"],
    }

    metrics_to_report = [
        "marginal_coverage",
        "worst_group_coverage",
        "avg_set_size",
        "set_size_disparity",
    ]

    for method_name, metric_list in methods.items():
        if not metric_list:
            continue
        method_summary = {}
        for m in metrics_to_report:
            vals = [t[m] for t in metric_list if m in t and t[m] is not None]
            if vals:
                mean = np.mean(vals)
                std = np.std(vals)
                method_summary[m] = f"{mean:.4f} +/- {std:.4f}"
            else:
                method_summary[m] = "N/A"
        summary[method_name] = method_summary

    return summary
