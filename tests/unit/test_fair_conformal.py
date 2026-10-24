import pytest
import numpy as np
from src.conformal.fair_conformal import FairTransCP

def test_fairtranscp_calibration():
    """Verify that FairTransCP adjusts quantiles based on fairness weight."""
    # Simple binary classification, 2 groups
    prob_matrix = np.array([
        [0.8, 0.2], [0.8, 0.2], [0.8, 0.2], # Group 0: confident
        [0.6, 0.4], [0.6, 0.4], [0.6, 0.4], # Group 1: less confident
    ])
    y_true = np.array([0, 0, 0, 0, 0, 0])
    sensitive_attr = np.array([0, 0, 0, 1, 1, 1])
    
    # Pure coverage equity (weight=0) -> quantiles should be different
    cp_equity = FairTransCP(alpha=0.1, fairness_weight=0.0)
    cp_equity.calibrate(prob_matrix, y_true, sensitive_attr)
    q_equity = cp_equity.get_diagnostics()["adjusted_quantiles"]
    
    # Pure set-size equity (weight=1) -> quantiles should be closer (marginal)
    cp_size = FairTransCP(alpha=0.1, fairness_weight=1.0)
    cp_size.calibrate(prob_matrix, y_true, sensitive_attr)
    q_size = cp_size.get_diagnostics()["adjusted_quantiles"]
    
    # Check that weight=1.0 results in more similar quantiles than weight=0.0
    diff_equity = abs(q_equity[0] - q_equity[1])
    diff_size = abs(q_size[0] - q_size[1])
    assert diff_size < diff_equity

def test_fairtranscp_prediction_sets():
    """Verify FairTransCP produces valid prediction sets."""
    prob_matrix = np.array([[0.9, 0.1], [0.1, 0.9]])
    y_true = np.array([0, 1])
    sensitive_attr = np.array([0, 1])
    
    cp = FairTransCP(alpha=0.1)
    cp.calibrate(prob_matrix, y_true, sensitive_attr)
    
    sets = cp.predict_sets(prob_matrix, sensitive_attr)
    assert len(sets) == 2
    assert isinstance(sets[0], list)
