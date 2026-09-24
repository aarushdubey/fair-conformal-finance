"""
Base conformal predictor classes.

Implements the split conformal prediction framework following
Vovk et al. (2005) and Lei et al. (2018). We keep things modular
so that different nonconformity scores can be plugged in easily.
"""

import numpy as np
from typing import Optional


class SplitConformalClassifier:
    """
    Split conformal classifier for multi-class problems.

    Given a pre-trained classifier and a held-out calibration set,
    this produces prediction sets with guaranteed marginal coverage.

    The key idea: instead of outputting a single predicted class,
    we output a *set* of classes that contains the true label with
    probability at least (1 - alpha). The magic is that this guarantee
    holds regardless of the data distribution, as long as the calibration
    data is exchangeable with the test data.

    Parameters
    ----------
    alpha : float
        Desired miscoverage rate, e.g. 0.1 for 90% coverage.
    score_fn : str
        Which nonconformity score to use. Options:
        - 'softmax': 1 - f(x)_y, where f(x)_y is the predicted
          probability for the true class y. Simple and effective.
        - 'aps': Adaptive Prediction Sets (Romano et al., 2020).
          Accumulates sorted probabilities until the true class is
          included. Tends to produce smaller sets on average.
    random_state : int or None
        Seed for reproducibility. We use this for tie-breaking
        in the quantile computation.
    """

    def __init__(self, alpha=0.1, score_fn="softmax", random_state=42):
        self.alpha = alpha
        self.score_fn = score_fn
        self.random_state = random_state

        # These get populated during calibration
        self._calibration_scores = None
        self._quantile = None
        self._n_classes = None

    def calibrate(self, prob_matrix, y_true):
        """
        Calibrate on a held-out set.

        We compute the nonconformity score for each calibration example
        and then find the (1 - alpha)-quantile. This quantile becomes
        our threshold for constructing prediction sets at test time.

        Parameters
        ----------
        prob_matrix : np.ndarray, shape (n_samples, n_classes)
            Predicted class probabilities from the base model.
        y_true : np.ndarray, shape (n_samples,)
            True labels for the calibration set.
        """
        n_cal = len(y_true)
        self._n_classes = prob_matrix.shape[1]

        # Compute nonconformity scores for each calibration example
        scores = self._compute_scores(prob_matrix, y_true)
        self._calibration_scores = scores

        # The finite-sample correction: we use ceil((n+1)(1-alpha))/n
        # quantile level, not just (1-alpha). This is what gives us
        # the marginal coverage guarantee.
        adjusted_level = np.ceil((n_cal + 1) * (1 - self.alpha)) / n_cal
        adjusted_level = min(adjusted_level, 1.0)

        self._quantile = np.quantile(scores, adjusted_level, method="higher")

    def predict_sets(self, prob_matrix):
        """
        Construct prediction sets for new examples.

        For each test point, we include class k in the prediction set
        if its nonconformity score s(x, k) is at most the calibration
        quantile. Lower scores mean the model is more "comfortable"
        with that class, so they get included.

        Parameters
        ----------
        prob_matrix : np.ndarray, shape (n_samples, n_classes)
            Predicted class probabilities for test examples.

        Returns
        -------
        prediction_sets : list of lists
            For each test example, the list of class labels included
            in the prediction set.
        """
        if self._quantile is None:
            raise RuntimeError(
                "Must call calibrate() before predict_sets(). "
                "Need calibration data to set the threshold."
            )

        n_test = prob_matrix.shape[0]
        prediction_sets = []

        for i in range(n_test):
            pset = []
            for k in range(self._n_classes):
                # Score for hypothetical label k
                score_k = self._compute_score_for_label(prob_matrix[i], k)
                if score_k <= self._quantile:
                    pset.append(k)

            # Edge case: empty set. Shouldn't happen often with proper
            # alpha, but just in case we include the most likely class.
            if len(pset) == 0:
                pset = [np.argmax(prob_matrix[i])]

            prediction_sets.append(pset)

        return prediction_sets

    def predict(self, prob_matrix):
        """Alias for predict_sets to maintain familiar scikit-learn style interface."""
        return self.predict_sets(prob_matrix)

    def _compute_scores(self, prob_matrix, y_true):
        """Compute nonconformity scores for labeled examples."""
        if self.score_fn == "softmax":
            return self._softmax_scores(prob_matrix, y_true)
        elif self.score_fn == "aps":
            return self._aps_scores(prob_matrix, y_true)
        else:
            raise ValueError(f"Unknown score function: {self.score_fn}")

    def _compute_score_for_label(self, prob_vector, label):
        """Compute the nonconformity score for a single (x, y) pair."""
        if self.score_fn == "softmax":
            return 1.0 - prob_vector[label]
        elif self.score_fn == "aps":
            return self._aps_score_single(prob_vector, label)
        else:
            raise ValueError(f"Unknown score function: {self.score_fn}")

    @staticmethod
    def _softmax_scores(prob_matrix, y_true):
        """
        Simple softmax-based score: s(x, y) = 1 - p_hat(y | x).

        Higher score means the model is less confident about the true
        class. This is the simplest nonconformity measure but works
        surprisingly well in practice.
        """
        n = len(y_true)
        scores = np.zeros(n)
        for i in range(n):
            scores[i] = 1.0 - prob_matrix[i, y_true[i]]
        return scores

    @staticmethod
    def _aps_scores(prob_matrix, y_true):
        """
        Adaptive Prediction Sets (Romano et al., 2020).

        The score is the cumulative probability mass when we sort
        classes by decreasing probability and accumulate until we
        include the true class. This typically gives tighter sets
        than the simple softmax score.

        Intuitively: if the true class has the highest predicted
        probability, its APS score equals just that probability.
        If it's ranked lower, we have to "use up" more probability
        mass to reach it, resulting in a higher score.
        """
        n = len(y_true)
        scores = np.zeros(n)

        for i in range(n):
            probs = prob_matrix[i]
            sorted_indices = np.argsort(-probs)  # descending order

            cumulative = 0.0
            for idx in sorted_indices:
                cumulative += probs[idx]
                if idx == y_true[i]:
                    scores[i] = cumulative
                    break

        return scores

    @staticmethod
    def _aps_score_single(prob_vector, label):
        """APS score for a single hypothetical (x, label) pair."""
        sorted_indices = np.argsort(-prob_vector)
        cumulative = 0.0
        for idx in sorted_indices:
            cumulative += prob_vector[idx]
            if idx == label:
                return cumulative
        # Should never reach here, but just in case
        return 1.0

    @property
    def calibration_scores(self):
        """Access the calibration scores (useful for diagnostics)."""
        return self._calibration_scores

    @property
    def threshold(self):
        """The calibrated quantile threshold."""
        return self._quantile
