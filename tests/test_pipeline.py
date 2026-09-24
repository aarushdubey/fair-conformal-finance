"""
Pipeline smoke test and sanity check for FairTransCP.

Verifies:
1. Dataset loading and preprocessing
2. Base classifier training
3. Split conformal prediction calibration and prediction
4. FairTransCP group-aware calibration
5. Fairness metric calculation
"""

import sys
import os
import numpy as np

# Ensure src is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.loaders import load_dataset
from src.models.classifiers import get_classifier
from src.conformal.base import SplitConformalClassifier
from src.conformal.fair_conformal import FairTransCP
from src.fairness.metrics import compute_all_metrics


def test_end_to_end_german_credit():
    print("--> 1. Loading German Credit dataset...")
    data = load_dataset("german_credit", sensitive="gender")
    X, y, sensitive = data["X"], data["y"], data["sensitive"]
    
    unique_groups, group_counts = np.unique(sensitive, return_counts=True)
    print(f"    Features shape: {X.shape}, Samples: {len(y)}")
    print(f"    Class distribution: {dict(zip(*np.unique(y, return_counts=True)))}")
    print(f"    Demographic groups: {dict(zip(unique_groups, group_counts))}")
    
    # 3-way split: train (50%), cal (25%), test (25%)
    n = len(X)
    rng = np.random.RandomState(42)
    indices = rng.permutation(n)
    
    n_train = int(0.5 * n)
    n_cal = int(0.25 * n)
    
    train_idx = indices[:n_train]
    cal_idx = indices[n_train:n_train + n_cal]
    test_idx = indices[n_train + n_cal:]
    
    X_train, y_train = X[train_idx], y[train_idx]
    X_cal, y_cal, s_cal = X[cal_idx], y[cal_idx], sensitive[cal_idx]
    X_test, y_test, s_test = X[test_idx], y[test_idx], sensitive[test_idx]
    
    print("--> 2. Fitting Random Forest model...")
    model = get_classifier("rf", random_state=42)
    model.fit(X_train, y_train)
    
    cal_probs = model.predict_proba(X_cal)
    test_probs = model.predict_proba(X_test)
    
    alpha = 0.10  # 90% target coverage
    
    print("--> 3. Evaluating Standard Conformal Prediction...")
    std_cp = SplitConformalClassifier(alpha=alpha, score_fn="softmax")
    std_cp.calibrate(cal_probs, y_cal)
    std_sets = std_cp.predict(test_probs)
    std_metrics = compute_all_metrics(std_sets, y_test, s_test)
    print(f"    Standard CP:")
    print(f"      Marginal Coverage    = {std_metrics['marginal_coverage']:.3f} (target >= {1 - alpha:.2f})")
    print(f"      Worst-Group Coverage = {std_metrics['worst_group_coverage']:.3f}")
    print(f"      Coverage Gap         = {std_metrics['coverage_gap']:.3f}")
    print(f"      Avg Set Size         = {std_metrics['avg_set_size']:.3f}")
    print(f"      Set-Size Disparity   = {std_metrics['set_size_disparity']:.3f}")
    
    print("--> 4. Evaluating FairTransCP (Proposed Framework)...")
    fair_cp = FairTransCP(alpha=alpha, fairness_weight=0.4, score_fn="softmax")
    fair_cp.calibrate(cal_probs, y_cal, s_cal)
    fair_sets = fair_cp.predict(test_probs, s_test)
    fair_metrics = compute_all_metrics(fair_sets, y_test, s_test)
    print(f"    FairTransCP:")
    print(f"      Marginal Coverage    = {fair_metrics['marginal_coverage']:.3f}")
    print(f"      Worst-Group Coverage = {fair_metrics['worst_group_coverage']:.3f}")
    print(f"      Coverage Gap         = {fair_metrics['coverage_gap']:.3f}")
    print(f"      Avg Set Size         = {fair_metrics['avg_set_size']:.3f}")
    print(f"      Set-Size Disparity   = {fair_metrics['set_size_disparity']:.3f}")
    
    print("\n[SUCCESS] Pipeline smoke test completed cleanly!")


if __name__ == "__main__":
    test_end_to_end_german_credit()
