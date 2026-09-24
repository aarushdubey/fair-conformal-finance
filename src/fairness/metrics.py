"""
Fairness metrics for conformal prediction sets.

We measure fairness along two axes:
1. Coverage equity — do all groups get similar coverage rates?
2. Set-size equity — do all groups get similarly-sized prediction sets?

The key insight motivating our paper is that these two objectives
can conflict. Equalizing coverage may require giving some groups
much larger (more ambiguous) prediction sets, which creates a
different kind of unfairness in practice.
"""

import numpy as np
from typing import Dict, List


def compute_group_coverage(
    prediction_sets: List[List[int]],
    y_true: np.ndarray,
    sensitive_attr: np.ndarray,
) -> Dict[str, float]:
    """
    Compute coverage rate for each demographic group.

    Coverage for a group is the fraction of examples where the
    true label is contained in the prediction set.

    Parameters
    ----------
    prediction_sets : list of lists
        The prediction set for each test example.
    y_true : np.ndarray
        True labels.
    sensitive_attr : np.ndarray
        Group membership.

    Returns
    -------
    dict mapping group_name -> coverage_rate
    """
    groups = np.unique(sensitive_attr)
    group_coverage = {}

    for g in groups:
        mask = sensitive_attr == g
        n_g = mask.sum()
        if n_g == 0:
            continue

        covered = 0
        indices = np.where(mask)[0]
        for i in indices:
            if y_true[i] in prediction_sets[i]:
                covered += 1

        group_coverage[str(g)] = covered / n_g

    return group_coverage


def compute_group_set_sizes(
    prediction_sets: List[List[int]],
    sensitive_attr: np.ndarray,
) -> Dict[str, float]:
    """
    Compute the average prediction set size for each group.

    Larger sets mean more ambiguity — the model is less decisive
    for that group. If one group consistently gets larger sets,
    that's a form of inequity even if coverage is equalized.
    """
    groups = np.unique(sensitive_attr)
    group_sizes = {}

    for g in groups:
        mask = sensitive_attr == g
        indices = np.where(mask)[0]
        if len(indices) == 0:
            continue

        sizes = [len(prediction_sets[i]) for i in indices]
        group_sizes[str(g)] = np.mean(sizes)

    return group_sizes


def coverage_gap(group_coverage: Dict[str, float]) -> float:
    """
    Maximum coverage difference between any two groups.

    This is the standard fairness metric for conformal prediction.
    A coverage gap of 0 means perfect coverage equity.
    In practice, gaps under 0.03 are considered acceptable.
    """
    values = list(group_coverage.values())
    if len(values) < 2:
        return 0.0
    return max(values) - min(values)


def set_size_disparity(group_sizes: Dict[str, float]) -> float:
    """
    Ratio of largest to smallest average set size across groups.

    A disparity of 1.0 means perfect set-size equity.
    Values much larger than 1.0 indicate that some groups face
    significantly more ambiguity than others.
    """
    values = list(group_sizes.values())
    if len(values) < 2:
        return 1.0
    min_size = min(values)
    max_size = max(values)
    if min_size == 0:
        return float("inf")
    return max_size / min_size


def worst_group_coverage(group_coverage: Dict[str, float]) -> float:
    """Minimum coverage across all groups."""
    if not group_coverage:
        return 0.0
    return min(group_coverage.values())


def marginal_coverage(
    prediction_sets: List[List[int]], y_true: np.ndarray
) -> float:
    """Overall coverage rate across all examples."""
    n = len(y_true)
    if n == 0:
        return 0.0
    covered = sum(
        1 for i in range(n) if y_true[i] in prediction_sets[i]
    )
    return covered / n


def average_set_size(prediction_sets: List[List[int]]) -> float:
    """Average prediction set size across all examples."""
    if not prediction_sets:
        return 0.0
    return np.mean([len(s) for s in prediction_sets])


def compute_all_metrics(
    prediction_sets: List[List[int]],
    y_true: np.ndarray,
    sensitive_attr: np.ndarray,
) -> Dict[str, float]:
    """
    Convenience function: compute all fairness metrics at once.

    Returns a flat dictionary suitable for logging to a results table.
    """
    gc = compute_group_coverage(prediction_sets, y_true, sensitive_attr)
    gs = compute_group_set_sizes(prediction_sets, sensitive_attr)

    return {
        "marginal_coverage": marginal_coverage(prediction_sets, y_true),
        "worst_group_coverage": worst_group_coverage(gc),
        "coverage_gap": coverage_gap(gc),
        "avg_set_size": average_set_size(prediction_sets),
        "set_size_disparity": set_size_disparity(gs),
        **{f"coverage_{k}": v for k, v in gc.items()},
        **{f"avg_size_{k}": v for k, v in gs.items()},
    }
