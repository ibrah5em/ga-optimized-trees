"""Unit tests for XGBoostBaseline export and basic behavior."""

import numpy as np
import pytest

from ga_trees.baselines import XGBoostBaseline


def _make_toy_data():
    """Simple binary classification dataset."""
    X = np.array([[0.0], [1.0], [0.0], [1.0]])
    y = np.array([0, 1, 0, 1])
    return X, y


def test_xgboostbaseline_importable():
    """XGBoostBaseline class must be importable from ga_trees.baselines."""
    assert XGBoostBaseline is not None


def test_xgboostbaseline_fit_predict_basic():
    """Fit and predict produce correct-shaped output when XGBoost is installed."""
    X, y = _make_toy_data()
    clf = XGBoostBaseline(max_depth=2, n_estimators=10, random_state=0)
    clf.fit(X, y)

    # If XGBoost isn't installed the implementation sets model to None
    if clf.model is None:
        pytest.skip("XGBoost not installed; skipping fit/predict behavior checks")

    preds = clf.predict(X)
    assert isinstance(preds, np.ndarray)
    assert preds.shape[0] == y.shape[0]

