"""
Wrappers for classical ML classifiers.

We deliberately restrict ourselves to tree-based ensemble methods.
The whole point is to show that you don't need deep learning to
get meaningful uncertainty quantification in finance — classical
models with conformal prediction can do it, with coverage guarantees
that neural networks can't easily provide.

Each wrapper provides a consistent interface: fit(), predict_proba(),
so that the conformal prediction module doesn't need to know which
model it's working with.
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import numpy as np

try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None

try:
    from lightgbm import LGBMClassifier
except ImportError:
    LGBMClassifier = None


def get_classifier(name, random_state=42):
    """
    Factory function — get a classifier by name with sensible defaults.

    We set conservative hyperparameters as defaults. For the paper
    we'll tune these via cross-validation, but these defaults should
    give reasonable results out of the box.

    Parameters
    ----------
    name : str
        One of 'rf', 'xgboost', 'lightgbm'.
    random_state : int
        For reproducibility.

    Returns
    -------
    sklearn-compatible classifier
    """
    if name == "rf":
        return RandomForestClassifier(
            n_estimators=300,
            max_depth=10,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        )

    elif name == "xgboost":
        if XGBClassifier is None:
            raise ImportError("xgboost not installed. pip install xgboost")
        return XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=random_state,
            n_jobs=-1,
            verbosity=0,
        )

    elif name == "lightgbm":
        if LGBMClassifier is None:
            raise ImportError("lightgbm not installed. pip install lightgbm")
        return LGBMClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=random_state,
            n_jobs=-1,
            verbose=-1,
        )

    else:
        available = ["rf", "xgboost", "lightgbm"]
        raise ValueError(f"Unknown model '{name}'. Options: {available}")


def tune_classifier(clf, X_train, y_train, param_grid=None, cv=5):
    """
    Quick hyperparameter tuning via cross-validation.

    We keep the search space small to avoid overfitting on the
    calibration set. The conformal guarantee holds regardless of
    how good the base model is, so we don't need to squeeze out
    every last bit of accuracy.

    Parameters
    ----------
    clf : classifier
        Unfitted classifier instance.
    X_train : np.ndarray
        Training features.
    y_train : np.ndarray
        Training labels.
    param_grid : dict or None
        Search grid. If None, uses a small default grid.
    cv : int
        Number of cross-validation folds.

    Returns
    -------
    best_clf : fitted classifier with best hyperparameters
    """
    if param_grid is None:
        # Minimal default grid — just depth and regularization
        param_grid = {
            "max_depth": [4, 6, 8],
            "n_estimators": [200, 300],
        }

    search = GridSearchCV(
        clf,
        param_grid,
        cv=cv,
        scoring="accuracy",
        n_jobs=-1,
        refit=True,
    )
    search.fit(X_train, y_train)

    return search.best_estimator_
