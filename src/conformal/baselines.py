"""
State-of-the-Art baselines for fair conformal prediction.

Implements:
1. MondrianCP: Independent group-conditional conformal calibration (Vovk et al., 2012).
2. LCCP: Label-Clustered Conformal Prediction (Liu et al., 2026).
3. GenericFairCP: Threshold search satisfying bounded coverage gap epsilon (Maneriker et al., ICLR 2025).
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from .base import SplitConformalClassifier


class MondrianCP:
    """
    Mondrian Conformal Prediction.
    Ensures group-conditional validity by calibrating independent empirical quantiles
    per sensitive demographic group.

    Reference: Vovk et al. (2012), Conditional Validity of Inductive Conformal Predictors.
    """
    def __init__(self, alpha: float = 0.1, score_fn: str = "softmax", random_state: int = 42):
        self.alpha = alpha
        self.score_fn = score_fn
        self.random_state = random_state
        self._group_thresholds: Dict[Any, float] = {}
        self._n_classes: Optional[int] = None

    def calibrate(self, prob_matrix: np.ndarray, y_true: np.ndarray, sensitive_attr: np.ndarray):
        self._n_classes = prob_matrix.shape[1]
        groups = np.unique(sensitive_attr)

        for g in groups:
            mask = (sensitive_attr == g)
            g_probs = prob_matrix[mask]
            g_labels = y_true[mask]

            cp = SplitConformalClassifier(
                alpha=self.alpha,
                score_fn=self.score_fn,
                random_state=self.random_state,
            )
            cp.calibrate(g_probs, g_labels)
            self._group_thresholds[g] = cp.threshold

    def predict_sets(self, prob_matrix: np.ndarray, sensitive_attr: np.ndarray) -> List[List[int]]:
        if not self._group_thresholds:
            raise RuntimeError("Call calibrate() first before predict_sets().")

        n_test = prob_matrix.shape[0]
        prediction_sets = []
        base_cp = SplitConformalClassifier(score_fn=self.score_fn)

        for i in range(n_test):
            g = sensitive_attr[i]
            threshold = self._group_thresholds.get(
                g, float(np.mean(list(self._group_thresholds.values())))
            )

            pset = []
            for k in range(self._n_classes):
                score_k = base_cp._compute_score_for_label(prob_matrix[i], k)
                if score_k <= threshold:
                    pset.append(k)

            if len(pset) == 0:
                pset = [int(np.argmax(prob_matrix[i]))]
            prediction_sets.append(pset)

        return prediction_sets


class LCCP:
    """
    Label-Clustered Conformal Prediction.
    Calibrates thresholds per difficulty cluster in label space to mitigate
    set-size disparity while maintaining label-conditional coverage.

    Reference: Liu et al. (2026).
    """
    def __init__(self, alpha: float = 0.1, score_fn: str = "softmax", random_state: int = 42):
        self.alpha = alpha
        self.score_fn = score_fn
        self.random_state = random_state
        self._cluster_thresholds: Dict[int, float] = {}
        self._label_to_cluster: Dict[int, int] = {}
        self._n_classes: Optional[int] = None

    def calibrate(self, prob_matrix: np.ndarray, y_true: np.ndarray, sensitive_attr: Optional[np.ndarray] = None):
        self._n_classes = prob_matrix.shape[1]
        base_cp = SplitConformalClassifier(score_fn=self.score_fn)
        
        all_scores = np.zeros((len(y_true), self._n_classes))
        for k in range(self._n_classes):
            for i in range(len(y_true)):
                all_scores[i, k] = base_cp._compute_score_for_label(prob_matrix[i], k)

        for k in range(self._n_classes):
            self._label_to_cluster[k] = k

        for c in range(self._n_classes):
            mask = (y_true == c)
            if mask.sum() == 0:
                continue

            c_scores = all_scores[mask, c]
            n_c = len(c_scores)
            level = np.ceil((n_c + 1) * (1 - self.alpha)) / n_c
            self._cluster_thresholds[c] = float(np.quantile(c_scores, min(level, 1.0), method="higher"))

    def predict_sets(self, prob_matrix: np.ndarray, sensitive_attr: Optional[np.ndarray] = None) -> List[List[int]]:
        if not self._cluster_thresholds:
            raise RuntimeError("Call calibrate() first before predict_sets().")

        n_test = prob_matrix.shape[0]
        prediction_sets = []
        base_cp = SplitConformalClassifier(score_fn=self.score_fn)

        for i in range(n_test):
            pset = []
            for k in range(self._n_classes):
                cluster = self._label_to_cluster[k]
                threshold = self._cluster_thresholds.get(cluster, 1.0)
                score_k = base_cp._compute_score_for_label(prob_matrix[i], k)
                if score_k <= threshold:
                    pset.append(k)

            if len(pset) == 0:
                pset = [int(np.argmax(prob_matrix[i]))]
            prediction_sets.append(pset)

        return prediction_sets


class GenericFairCP:
    """
    Generic Conformal Fairness (CF) Framework.
    Finds a single global threshold that constrains maximum demographic coverage gap <= epsilon
    while satisfying nominal marginal coverage >= 1 - alpha.

    Reference: Maneriker et al. (ICLR 2025).
    """
    def __init__(self, alpha: float = 0.1, epsilon: float = 0.03, score_fn: str = "softmax", random_state: int = 42):
        self.alpha = alpha
        self.epsilon = epsilon
        self.score_fn = score_fn
        self.random_state = random_state
        self._optimal_threshold: Optional[float] = None
        self._n_classes: Optional[int] = None

    def calibrate(self, prob_matrix: np.ndarray, y_true: np.ndarray, sensitive_attr: np.ndarray):
        self._n_classes = prob_matrix.shape[1]
        base_cp = SplitConformalClassifier(score_fn=self.score_fn)
        scores = base_cp._compute_scores(prob_matrix, y_true)

        possible_thresholds = np.linspace(np.min(scores), np.max(scores), 100)
        best_threshold = float(np.max(scores))

        for t in possible_thresholds:
            groups = np.unique(sensitive_attr)
            covs = []
            for g in groups:
                mask = (sensitive_attr == g)
                g_scores = scores[mask]
                covs.append(np.mean(g_scores <= t))

            gap = max(covs) - min(covs)
            overall_cov = np.mean(scores <= t)

            if gap <= self.epsilon and overall_cov >= (1 - self.alpha):
                best_threshold = float(t)
                break

        self._optimal_threshold = best_threshold

    def predict_sets(self, prob_matrix: np.ndarray, sensitive_attr: Optional[np.ndarray] = None) -> List[List[int]]:
        if self._optimal_threshold is None:
            raise RuntimeError("Call calibrate() first before predict_sets().")

        n_test = prob_matrix.shape[0]
        prediction_sets = []
        base_cp = SplitConformalClassifier(score_fn=self.score_fn)

        for i in range(n_test):
            pset = []
            for k in range(self._n_classes):
                score_k = base_cp._compute_score_for_label(prob_matrix[i], k)
                if score_k <= self._optimal_threshold:
                    pset.append(k)

            if len(pset) == 0:
                pset = [int(np.argmax(prob_matrix[i]))]
            prediction_sets.append(pset)

        return prediction_sets
