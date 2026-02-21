"""Unit tests for src/ga_trees/evaluation/metrics.py.

Covers:
- MetricsCalculator.calculate_classification_metrics
- MetricsCalculator.calculate_interpretability_metrics
"""

import numpy as np
import pytest

from ga_trees.evaluation.metrics import MetricsCalculator
from ga_trees.fitness.calculator import FitnessCalculator, TreePredictor
from ga_trees.genotype.tree_genotype import (
    TreeGenotype,
    create_internal_node,
    create_leaf_node,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _perfect_predictions():
    """Labels and predictions that are identical → accuracy = 1.0."""
    y_true = np.array([0, 1, 0, 1, 2, 2, 0])
    y_pred = y_true.copy()
    return y_true, y_pred


def _binary_predictions():
    """Binary classification with one mismatch → accuracy = 4/5 = 0.8."""
    y_true = np.array([0, 1, 0, 1, 0])
    y_pred = np.array([0, 1, 1, 1, 0])  # index 2 is wrong
    return y_true, y_pred


def _tree_with_interpretability():
    """Return a tree that has had interpretability_ set via FitnessCalculator."""
    left = create_leaf_node(0, depth=1)
    right = create_leaf_node(1, depth=1)
    root = create_internal_node(0, 0.5, left, right, depth=0)
    tree = TreeGenotype(root=root, n_features=4, n_classes=2, max_depth=5)

    X = np.array([[0.2, 0.5, 0.5, 0.5], [0.8, 0.5, 0.5, 0.5]])
    y = np.array([0, 1])
    FitnessCalculator().calculate_fitness(tree, X, y)
    return tree


# ---------------------------------------------------------------------------
# Classification metrics tests
# ---------------------------------------------------------------------------


class TestCalculateClassificationMetrics:
    """Tests for MetricsCalculator.calculate_classification_metrics."""

    def test_returns_dict(self):
        y_true, y_pred = _binary_predictions()
        result = MetricsCalculator.calculate_classification_metrics(y_true, y_pred)
        assert isinstance(result, dict)

    def test_required_keys_present(self):
        y_true, y_pred = _binary_predictions()
        result = MetricsCalculator.calculate_classification_metrics(y_true, y_pred)
        required = {
            "accuracy",
            "precision_macro",
            "precision_weighted",
            "recall_macro",
            "recall_weighted",
            "f1_macro",
            "f1_weighted",
            "confusion_matrix",
        }
        assert required.issubset(result.keys()), f"Missing keys: {required - result.keys()}"

    def test_perfect_accuracy(self):
        y_true, y_pred = _perfect_predictions()
        result = MetricsCalculator.calculate_classification_metrics(y_true, y_pred)
        assert result["accuracy"] == pytest.approx(1.0)

    def test_imperfect_accuracy(self):
        y_true, y_pred = _binary_predictions()
        result = MetricsCalculator.calculate_classification_metrics(y_true, y_pred)
        assert result["accuracy"] == pytest.approx(0.8)

    def test_accuracy_range(self):
        y_true = np.array([0, 1, 0, 1, 1])
        y_pred = np.array([1, 0, 1, 0, 0])  # all wrong
        result = MetricsCalculator.calculate_classification_metrics(y_true, y_pred)
        assert 0.0 <= result["accuracy"] <= 1.0

    def test_confusion_matrix_is_list(self):
        y_true, y_pred = _binary_predictions()
        result = MetricsCalculator.calculate_classification_metrics(y_true, y_pred)
        assert isinstance(result["confusion_matrix"], list)

    def test_confusion_matrix_shape_binary(self):
        y_true, y_pred = _binary_predictions()
        result = MetricsCalculator.calculate_classification_metrics(y_true, y_pred)
        cm = result["confusion_matrix"]
        assert len(cm) == 2 and len(cm[0]) == 2

    def test_confusion_matrix_shape_multiclass(self):
        y_true = np.array([0, 1, 2, 0, 1, 2])
        y_pred = np.array([0, 2, 1, 0, 0, 2])
        result = MetricsCalculator.calculate_classification_metrics(y_true, y_pred)
        cm = result["confusion_matrix"]
        assert len(cm) == 3 and len(cm[0]) == 3

    def test_f1_weighted_perfect(self):
        y_true, y_pred = _perfect_predictions()
        result = MetricsCalculator.calculate_classification_metrics(y_true, y_pred)
        assert result["f1_weighted"] == pytest.approx(1.0)

    def test_precision_macro_range(self):
        y_true, y_pred = _binary_predictions()
        result = MetricsCalculator.calculate_classification_metrics(y_true, y_pred)
        assert 0.0 <= result["precision_macro"] <= 1.0

    def test_recall_macro_range(self):
        y_true, y_pred = _binary_predictions()
        result = MetricsCalculator.calculate_classification_metrics(y_true, y_pred)
        assert 0.0 <= result["recall_macro"] <= 1.0

    def test_roc_auc_not_present_without_probs(self):
        """roc_auc should not appear when y_prob is not provided."""
        y_true, y_pred = _binary_predictions()
        result = MetricsCalculator.calculate_classification_metrics(y_true, y_pred)
        assert "roc_auc" not in result

    def test_roc_auc_present_with_binary_probs(self):
        """roc_auc should appear for binary classification when y_prob is provided."""
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 1, 1])
        y_prob = np.array([[0.9, 0.1], [0.8, 0.2], [0.2, 0.8], [0.1, 0.9]])
        result = MetricsCalculator.calculate_classification_metrics(y_true, y_pred, y_prob)
        assert "roc_auc" in result
        assert result["roc_auc"] == pytest.approx(1.0)

    def test_roc_auc_absent_for_multiclass_with_probs(self):
        """roc_auc is skipped for multiclass even when y_prob is provided."""
        y_true = np.array([0, 1, 2, 0, 1, 2])
        y_pred = np.array([0, 1, 2, 0, 1, 2])
        y_prob = np.eye(3)[[0, 1, 2, 0, 1, 2]]  # one-hot
        result = MetricsCalculator.calculate_classification_metrics(y_true, y_pred, y_prob)
        # Implementation only computes roc_auc for binary (2 unique classes)
        assert "roc_auc" not in result

    def test_single_class_prediction_does_not_raise(self):
        """All predictions identical should not raise (zero_division=0 is set)."""
        y_true = np.array([0, 1, 1, 0])
        y_pred = np.array([1, 1, 1, 1])
        result = MetricsCalculator.calculate_classification_metrics(y_true, y_pred)
        assert "precision_macro" in result


# ---------------------------------------------------------------------------
# Interpretability metrics tests
# ---------------------------------------------------------------------------


class TestCalculateInterpretabilityMetrics:
    """Tests for MetricsCalculator.calculate_interpretability_metrics."""

    @pytest.fixture
    def fitted_tree(self):
        return _tree_with_interpretability()

    def test_returns_dict(self, fitted_tree):
        result = MetricsCalculator.calculate_interpretability_metrics(fitted_tree)
        assert isinstance(result, dict)

    def test_required_keys_present(self, fitted_tree):
        result = MetricsCalculator.calculate_interpretability_metrics(fitted_tree)
        required = {
            "depth",
            "num_nodes",
            "num_leaves",
            "features_used",
            "tree_balance",
            "interpretability_score",
        }
        assert required.issubset(result.keys())

    def test_depth_matches_tree(self, fitted_tree):
        result = MetricsCalculator.calculate_interpretability_metrics(fitted_tree)
        assert result["depth"] == fitted_tree.get_depth()

    def test_num_nodes_matches_tree(self, fitted_tree):
        result = MetricsCalculator.calculate_interpretability_metrics(fitted_tree)
        assert result["num_nodes"] == fitted_tree.get_num_nodes()

    def test_num_leaves_matches_tree(self, fitted_tree):
        result = MetricsCalculator.calculate_interpretability_metrics(fitted_tree)
        assert result["num_leaves"] == fitted_tree.get_num_leaves()

    def test_features_used_matches_tree(self, fitted_tree):
        result = MetricsCalculator.calculate_interpretability_metrics(fitted_tree)
        assert result["features_used"] == fitted_tree.get_num_features_used()

    def test_tree_balance_range(self, fitted_tree):
        result = MetricsCalculator.calculate_interpretability_metrics(fitted_tree)
        assert 0.0 <= result["tree_balance"] <= 1.0

    def test_interpretability_score_range(self, fitted_tree):
        score = MetricsCalculator.calculate_interpretability_metrics(fitted_tree)[
            "interpretability_score"
        ]
        assert 0.0 <= float(score) <= 1.0

    def test_depth1_tree_values(self):
        """Depth-1 tree: depth=1, nodes=3, leaves=2."""
        left = create_leaf_node(0, depth=1)
        right = create_leaf_node(1, depth=1)
        root = create_internal_node(0, 0.5, left, right, depth=0)
        tree = TreeGenotype(root=root, n_features=4, n_classes=2)
        X = np.array([[0.2, 0.0, 0.0, 0.0], [0.8, 0.0, 0.0, 0.0]])
        y = np.array([0, 1])
        FitnessCalculator().calculate_fitness(tree, X, y)

        result = MetricsCalculator.calculate_interpretability_metrics(tree)
        assert result["depth"] == 1
        assert result["num_nodes"] == 3
        assert result["num_leaves"] == 2
        assert result["features_used"] == 1
