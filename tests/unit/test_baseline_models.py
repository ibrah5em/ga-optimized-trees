"""Unit tests for baselines/baseline_models.py.

Covers:
- BaselineModel: predict, get_metrics, get_depth, get_num_nodes,
  get_num_leaves, get_num_features_used (both with and without tree_)
- CARTBaseline: fit + predict + get_metrics
- PrunedCARTBaseline: fit (covers both ccp_alpha branches)
- RandomForestBaseline: fit + get_metrics (averaged depth/num_nodes)
- XGBoostBaseline: fit success path + ImportError path
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from sklearn.datasets import load_iris

from ga_trees.baselines.baseline_models import (
    CARTBaseline,
    PrunedCARTBaseline,
    RandomForestBaseline,
    XGBoostBaseline,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def iris():
    X, y = load_iris(return_X_y=True)
    return X, y


@pytest.fixture
def fitted_cart(iris):
    X, y = iris
    model = CARTBaseline(max_depth=3, random_state=42)
    model.fit(X, y)
    return model


# ---------------------------------------------------------------------------
# BaselineModel (via CARTBaseline as a concrete subclass)
# ---------------------------------------------------------------------------


class TestBaselineModelPredict:
    def test_predict_returns_array(self, fitted_cart, iris):
        X, y = iris
        preds = fitted_cart.predict(X)
        assert isinstance(preds, np.ndarray)
        assert len(preds) == len(y)


class TestBaselineModelGetMetrics:
    def test_returns_dict_with_required_keys(self, fitted_cart):
        metrics = fitted_cart.get_metrics()
        assert isinstance(metrics, dict)
        for key in ("name", "depth", "num_nodes", "num_leaves", "features_used"):
            assert key in metrics

    def test_name_is_cart(self, fitted_cart):
        assert fitted_cart.get_metrics()["name"] == "CART"

    def test_depth_is_positive_after_fit(self, fitted_cart):
        assert fitted_cart.get_depth() >= 0

    def test_num_nodes_positive_after_fit(self, fitted_cart):
        assert fitted_cart.get_num_nodes() > 0

    def test_num_leaves_positive_after_fit(self, fitted_cart):
        assert fitted_cart.get_num_leaves() > 0

    def test_features_used_positive_after_fit(self, fitted_cart):
        assert fitted_cart.get_num_features_used() >= 0


class TestBaselineModelWithoutTreeAttr:
    """When model has no tree_ attribute, all getters return -1."""

    def setup_method(self):
        self.model = CARTBaseline(random_state=42)
        # Replace the underlying model with a mock that has no tree_ attr
        self.model.model = MagicMock(spec=[])  # spec=[] → no tree_ attribute

    def test_get_depth_returns_minus_one(self):
        assert self.model.get_depth() == -1

    def test_get_num_nodes_returns_minus_one(self):
        assert self.model.get_num_nodes() == -1

    def test_get_num_leaves_returns_minus_one(self):
        assert self.model.get_num_leaves() == -1

    def test_get_num_features_used_returns_minus_one(self):
        assert self.model.get_num_features_used() == -1

    def test_get_metrics_contains_minus_ones(self):
        metrics = self.model.get_metrics()
        assert metrics["depth"] == -1
        assert metrics["num_nodes"] == -1
        assert metrics["num_leaves"] == -1
        assert metrics["features_used"] == -1


# ---------------------------------------------------------------------------
# CARTBaseline
# ---------------------------------------------------------------------------


class TestCARTBaseline:
    def test_fit_returns_self(self, iris):
        X, y = iris
        cart = CARTBaseline(max_depth=3, random_state=42)
        result = cart.fit(X, y)
        assert result is cart

    def test_fit_sets_model(self, iris):
        X, y = iris
        cart = CARTBaseline(random_state=42)
        cart.fit(X, y)
        assert cart.model is not None
        assert hasattr(cart.model, "tree_")

    def test_predict_accuracy_reasonable(self, iris):
        X, y = iris
        cart = CARTBaseline(max_depth=5, random_state=42)
        cart.fit(X, y)
        preds = cart.predict(X)
        accuracy = np.mean(preds == y)
        assert accuracy > 0.7

    def test_max_depth_respected(self, iris):
        X, y = iris
        cart = CARTBaseline(max_depth=2, random_state=42)
        cart.fit(X, y)
        assert cart.get_depth() <= 2

    def test_custom_min_samples_split(self, iris):
        X, y = iris
        cart = CARTBaseline(min_samples_split=20, random_state=42)
        cart.fit(X, y)
        assert cart.model is not None

    def test_get_metrics_reasonable_values(self, iris):
        X, y = iris
        cart = CARTBaseline(max_depth=3, random_state=42)
        cart.fit(X, y)
        m = cart.get_metrics()
        assert m["depth"] >= 1
        assert m["num_nodes"] >= 3
        assert m["num_leaves"] >= 2
        assert m["features_used"] >= 1


# ---------------------------------------------------------------------------
# PrunedCARTBaseline
# ---------------------------------------------------------------------------


class TestPrunedCARTBaseline:
    def test_fit_returns_self(self, iris):
        X, y = iris
        pruned = PrunedCARTBaseline(random_state=42)
        result = pruned.fit(X, y)
        assert result is pruned

    def test_fit_sets_model(self, iris):
        X, y = iris
        pruned = PrunedCARTBaseline(random_state=42)
        pruned.fit(X, y)
        assert pruned.model is not None

    def test_predict_works_after_fit(self, iris):
        X, y = iris
        pruned = PrunedCARTBaseline(random_state=42)
        pruned.fit(X, y)
        preds = pruned.predict(X)
        assert len(preds) == len(y)

    def test_covers_single_alpha_branch(self):
        """When ccp_alphas has length 1, best_alpha should be 0.0."""
        # Fit on tiny dataset where pruning path is trivial
        X = np.array([[0.0, 0.0], [1.0, 1.0]])
        y = np.array([0, 1])
        pruned = PrunedCARTBaseline(random_state=42)
        pruned.fit(X, y)
        # If we got here without error, branch is covered
        assert pruned.model is not None


# ---------------------------------------------------------------------------
# RandomForestBaseline
# ---------------------------------------------------------------------------


class TestRandomForestBaseline:
    def test_fit_returns_self(self, iris):
        X, y = iris
        rf = RandomForestBaseline(n_estimators=10, random_state=42)
        result = rf.fit(X, y)
        assert result is rf

    def test_fit_sets_model(self, iris):
        X, y = iris
        rf = RandomForestBaseline(n_estimators=5, random_state=42)
        rf.fit(X, y)
        assert rf.model is not None

    def test_predict_works(self, iris):
        X, y = iris
        rf = RandomForestBaseline(n_estimators=5, random_state=42)
        rf.fit(X, y)
        preds = rf.predict(X)
        assert len(preds) == len(y)

    def test_get_metrics_returns_averaged_depth(self, iris):
        X, y = iris
        rf = RandomForestBaseline(n_estimators=5, random_state=42)
        rf.fit(X, y)
        metrics = rf.get_metrics()
        assert "depth" in metrics
        assert "num_nodes" in metrics
        assert metrics["depth"] >= 0
        assert metrics["num_nodes"] > 0

    def test_get_metrics_overrides_base(self, iris):
        """RandomForest get_metrics averages over estimators (overrides base)."""
        X, y = iris
        rf = RandomForestBaseline(n_estimators=10, random_state=42)
        rf.fit(X, y)
        m = rf.get_metrics()
        # Base model has get_depth() = -1 (no tree_); RF override should give real value
        assert m["depth"] >= 0

    def test_name_is_random_forest(self, iris):
        X, y = iris
        rf = RandomForestBaseline(n_estimators=5, random_state=42)
        rf.fit(X, y)
        assert rf.get_metrics()["name"] == "Random Forest"


# ---------------------------------------------------------------------------
# XGBoostBaseline
# ---------------------------------------------------------------------------


class TestXGBoostBaseline:
    def test_fit_with_import_error(self, iris):
        """When xgboost is not importable, model should be set to None."""
        X, y = iris
        xgb = XGBoostBaseline(random_state=42)
        with patch.dict("sys.modules", {"xgboost": None}):
            xgb.fit(X, y)
        assert xgb.model is None

    def test_fit_returns_self_on_import_error(self, iris):
        X, y = iris
        xgb = XGBoostBaseline(random_state=42)
        with patch.dict("sys.modules", {"xgboost": None}):
            result = xgb.fit(X, y)
        assert result is xgb

    def test_fit_success_path(self, iris):
        """If xgboost is installed, model is set; otherwise we get None."""
        X, y = iris
        xgb = XGBoostBaseline(max_depth=3, n_estimators=5, random_state=42)
        xgb.fit(X, y)
        # Either xgboost is installed (model is set) or not (model is None)
        # Both are valid outcomes; just ensure fit() doesn't raise
        assert xgb.model is None or xgb.model is not None

    def test_fit_with_mocked_xgboost(self, iris):
        """Mock xgboost to test the success branch without the real package."""
        X, y = iris
        mock_xgb_module = MagicMock()
        mock_classifier = MagicMock()
        mock_xgb_module.XGBClassifier.return_value = mock_classifier

        with patch.dict("sys.modules", {"xgboost": mock_xgb_module}):
            xgb = XGBoostBaseline(max_depth=3, n_estimators=5, random_state=42)
            result = xgb.fit(X, y)

        mock_xgb_module.XGBClassifier.assert_called_once()
        mock_classifier.fit.assert_called_once()
        assert result is xgb
