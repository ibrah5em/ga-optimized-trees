"""Unit tests for evaluation/explainability.py.

Covers:
- explain_with_shap when SHAP_AVAILABLE=False (print + return None)
- explain_with_shap when SHAP_AVAILABLE=True (mocked shap)
- explain_with_lime when LIME_AVAILABLE=False (print + return None)
- explain_with_lime when LIME_AVAILABLE=True (mocked lime)
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

import ga_trees.evaluation.explainability as expl_module
from ga_trees.evaluation.explainability import TreeExplainer
from ga_trees.genotype.tree_genotype import TreeGenotype, create_internal_node, create_leaf_node

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def simple_tree():
    left = create_leaf_node(0, depth=1)
    right = create_leaf_node(1, depth=1)
    root = create_internal_node(0, 0.5, left, right, depth=0)
    return TreeGenotype(root=root, n_features=4, n_classes=2)


@pytest.fixture
def small_X():
    np.random.seed(42)
    return np.random.rand(10, 4)


# ---------------------------------------------------------------------------
# explain_with_shap — unavailable path
# ---------------------------------------------------------------------------


class TestExplainWithShapUnavailable:
    """explain_with_shap when SHAP_AVAILABLE is False."""

    def test_prints_message(self, simple_tree, small_X, capsys):
        with patch.object(expl_module, "SHAP_AVAILABLE", False):
            TreeExplainer.explain_with_shap(simple_tree, small_X)
        captured = capsys.readouterr()
        assert "SHAP" in captured.out

    def test_returns_none(self, simple_tree, small_X):
        with patch.object(expl_module, "SHAP_AVAILABLE", False):
            result = TreeExplainer.explain_with_shap(simple_tree, small_X)
        assert result is None


# ---------------------------------------------------------------------------
# explain_with_shap — available path
# ---------------------------------------------------------------------------


class TestExplainWithShapAvailable:
    """explain_with_shap when SHAP_AVAILABLE is True (mocked shap)."""

    def _make_mock_shap(self):
        mock_shap = MagicMock()
        mock_explainer_instance = MagicMock()
        mock_shap_values = MagicMock()
        mock_explainer_instance.return_value = mock_shap_values
        mock_shap.Explainer.return_value = mock_explainer_instance
        return mock_shap, mock_explainer_instance, mock_shap_values

    def test_calls_explainer(self, simple_tree, small_X):
        mock_shap, mock_inst, mock_vals = self._make_mock_shap()
        with patch.object(expl_module, "SHAP_AVAILABLE", True), patch.object(
            expl_module, "shap", mock_shap, create=True
        ):
            TreeExplainer.explain_with_shap(simple_tree, small_X)
        mock_shap.Explainer.assert_called_once()

    def test_calls_summary_plot(self, simple_tree, small_X):
        mock_shap, mock_inst, mock_vals = self._make_mock_shap()
        with patch.object(expl_module, "SHAP_AVAILABLE", True), patch.object(
            expl_module, "shap", mock_shap, create=True
        ):
            TreeExplainer.explain_with_shap(simple_tree, small_X)
        mock_shap.summary_plot.assert_called_once()

    def test_passes_feature_names_to_summary_plot(self, simple_tree, small_X):
        mock_shap, mock_inst, mock_vals = self._make_mock_shap()
        feature_names = ["f0", "f1", "f2", "f3"]
        with patch.object(expl_module, "SHAP_AVAILABLE", True), patch.object(
            expl_module, "shap", mock_shap, create=True
        ):
            TreeExplainer.explain_with_shap(simple_tree, small_X, feature_names=feature_names)
        call_kwargs = mock_shap.summary_plot.call_args[1]
        assert call_kwargs.get("feature_names") == feature_names

    def test_does_not_print_unavailable_message(self, simple_tree, small_X, capsys):
        mock_shap, _, _ = self._make_mock_shap()
        with patch.object(expl_module, "SHAP_AVAILABLE", True), patch.object(
            expl_module, "shap", mock_shap, create=True
        ):
            TreeExplainer.explain_with_shap(simple_tree, small_X)
        captured = capsys.readouterr()
        assert "not available" not in captured.out.lower()


# ---------------------------------------------------------------------------
# explain_with_lime — unavailable path
# ---------------------------------------------------------------------------


class TestExplainWithLimeUnavailable:
    """explain_with_lime when LIME_AVAILABLE is False."""

    def test_prints_message(self, simple_tree, small_X, capsys):
        with patch.object(expl_module, "LIME_AVAILABLE", False):
            TreeExplainer.explain_with_lime(simple_tree, small_X, instance_idx=0)
        captured = capsys.readouterr()
        assert "LIME" in captured.out

    def test_returns_none(self, simple_tree, small_X):
        with patch.object(expl_module, "LIME_AVAILABLE", False):
            result = TreeExplainer.explain_with_lime(simple_tree, small_X, instance_idx=0)
        assert result is None


# ---------------------------------------------------------------------------
# explain_with_lime — available path
# ---------------------------------------------------------------------------


class TestExplainWithLimeAvailable:
    """explain_with_lime when LIME_AVAILABLE is True (mocked lime)."""

    def _make_mock_lime(self):
        mock_lime = MagicMock()
        mock_lime_tabular = MagicMock()
        mock_lime.lime_tabular = mock_lime_tabular
        mock_explainer = MagicMock()
        mock_lime_tabular.LimeTabularExplainer.return_value = mock_explainer
        mock_exp = MagicMock()
        mock_explainer.explain_instance.return_value = mock_exp
        return mock_lime, mock_explainer, mock_exp

    def test_creates_tabular_explainer(self, simple_tree, small_X):
        mock_lime, mock_expl, mock_exp = self._make_mock_lime()
        with patch.object(expl_module, "LIME_AVAILABLE", True), patch.object(
            expl_module, "lime", mock_lime, create=True
        ):
            TreeExplainer.explain_with_lime(simple_tree, small_X, instance_idx=0)
        mock_lime.lime_tabular.LimeTabularExplainer.assert_called_once()

    def test_calls_explain_instance(self, simple_tree, small_X):
        mock_lime, mock_expl, mock_exp = self._make_mock_lime()
        with patch.object(expl_module, "LIME_AVAILABLE", True), patch.object(
            expl_module, "lime", mock_lime, create=True
        ):
            TreeExplainer.explain_with_lime(simple_tree, small_X, instance_idx=2)
        mock_expl.explain_instance.assert_called_once()

    def test_calls_show_in_notebook(self, simple_tree, small_X):
        mock_lime, mock_expl, mock_exp = self._make_mock_lime()
        with patch.object(expl_module, "LIME_AVAILABLE", True), patch.object(
            expl_module, "lime", mock_lime, create=True
        ):
            TreeExplainer.explain_with_lime(simple_tree, small_X, instance_idx=0)
        mock_exp.show_in_notebook.assert_called_once()

    def test_passes_feature_names_to_explainer(self, simple_tree, small_X):
        mock_lime, mock_expl, mock_exp = self._make_mock_lime()
        feature_names = ["f0", "f1", "f2", "f3"]
        with patch.object(expl_module, "LIME_AVAILABLE", True), patch.object(
            expl_module, "lime", mock_lime, create=True
        ):
            TreeExplainer.explain_with_lime(
                simple_tree, small_X, instance_idx=0, feature_names=feature_names
            )
        call_kwargs = mock_lime.lime_tabular.LimeTabularExplainer.call_args[1]
        assert call_kwargs.get("feature_names") == feature_names

    def test_does_not_print_unavailable_message(self, simple_tree, small_X, capsys):
        mock_lime, _, _ = self._make_mock_lime()
        with patch.object(expl_module, "LIME_AVAILABLE", True), patch.object(
            expl_module, "lime", mock_lime, create=True
        ):
            TreeExplainer.explain_with_lime(simple_tree, small_X, instance_idx=0)
        captured = capsys.readouterr()
        assert "not available" not in captured.out.lower()
