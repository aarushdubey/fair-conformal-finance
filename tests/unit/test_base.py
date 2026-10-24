import pytest
import numpy as np
from src.conformal.base import SplitConformalClassifier

def test_split_conformal_calibration():
    """Verify that calibration correctly sets the quantile."""
    # 10 samples, 2 classes
    prob_matrix = np.array([
        [0.8, 0.2], [0.7, 0.3], [0.9, 0.1], [0.6, 0.4], [0.8, 0.2],
        [0.7, 0.3], [0.9, 0.1], [0.6, 0.4], [0.8, 0.2], [0.7, 0.3]
    ])
    y_true = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0]) # All class 0
    
    cp = SplitConformalClassifier(alpha=0.1, score_fn="softmax")
    cp.calibrate(prob_matrix, y_true)
    
    # Scores for softmax: 1 - prob(true_class)
    # scores = [0.2, 0.3, 0.1, 0.4, 0.2, 0.3, 0.1, 0.4, 0.2, 0.3]
    # sorted = [0.1, 0.1, 0.2, 0.2, 0.2, 0.3, 0.3, 0.3, 0.4, 0.4]
    # adjusted_level = ceil(11 * 0.9) / 10 = 10/10 = 1.0
    # quantile = sorted[9] = 0.4
    assert cp.threshold == 0.4

def test_split_conformal_prediction():
    """Verify that prediction sets contain the true label with high prob."""
    prob_matrix = np.array([[0.9, 0.1], [0.1, 0.9]])
    y_true = np.array([0, 1])
    
    cp = SplitConformalClassifier(alpha=0.1, score_fn="softmax")
    cp.calibrate(prob_matrix, y_true)
    
    # Test sets
    test_probs = np.array([[0.8, 0.2], [0.2, 0.8]])
    sets = cp.predict_sets(test_probs)
    
    assert 0 in sets[0]
    assert 1 in sets[1]

def test_aps_score_logic():
    """Verify the Adaptive Prediction Sets (APS) score computation."""
    cp = SplitConformalClassifier(score_fn="aps")
    # Class 0 is true, prob is 0.3, but class 1 is 0.7
    # sorted probs: [0.7, 0.3]. Accumulate: 0.7 (class 1), 0.7+0.3=1.0 (class 0)
    # APS score = 1.0
    score = cp._aps_score_single(np.array([0.3, 0.7]), 0)
    assert score == 1.0
    
    # Class 0 is true, prob is 0.7
    # sorted: [0.7, 0.3]. Accumulate: 0.7 (class 0)
    # APS score = 0.7
    score = cp._aps_score_single(np.array([0.7, 0.3]), 0)
    assert score == 0.7
