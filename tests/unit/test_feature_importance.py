"""Unit tests for src/ga_trees/evaluation/feature_importance.py.

Covers:
- FeatureImportanceAnalyzer.calculate_feature_frequency
- FeatureImportanceAnalyzer.calculate_feature_depth_importance
"""

import numpy as np
import pytest

from ga_trees.evaluation.feature_importance import FeatureImportanceAnalyzer
from ga_trees.genotype.tree_genotype import (
    TreeGenotype,
    create_internal_node,
    create_leaf_node,
)


# ---------------------------------------------------------------------------
# Tree factories
# ---------------------------------------------------------------------------


def _leaf_only_tree():
    """Single leaf node — no internal nodes, no features used."""
    root = create_leaf_node(prediction=0, depth=0)
    return TreeGenotype(root=root, n_features=4, n_classes=2)


def _single_split_tree(feature_idx: int = 0):
    """Depth-1 tree splitting on one feature."""
    left = create_leaf_node(0, depth=1)
    right = create_leaf_node(1, depth=1)
    root = create_internal_node(feature_idx, 0.5, left, right, depth=0)
    return TreeGenotype(root=root, n_features=4, n_classes=2)


def _two_feature_tree():
    """Depth-2 tree using features 0 (root) and 1 (one child).

    Structure:
        root (feat 0)
        ├── left: internal (feat 1)
        │   ├── leaf 0
        │   └── leaf 1
        └── right: leaf 2
    """
    ll = create_leaf_node(0, depth=2)
    lr = create_leaf_node(1, depth=2)
    left_internal = create_internal_node(1, 0.3, ll, lr, depth=1)
    right_leaf = create_leaf_node(2, depth=1)
    root = create_internal_node(0, 0.5, left_internal, right_leaf, depth=0)
    return TreeGenotype(root=root, n_features=4, n_classes=3)


def _repeated_feature_tree():
    """Depth-2 full tree where all internal nodes use feature 0.

    Structure:
        root (feat 0, depth 0)
        ├── left (feat 0, depth 1)
        │   ├── leaf 0
        │   └── leaf 1
        └── right (feat 0, depth 1)
            ├── leaf 0
            └── leaf 1
    """
    ll = create_leaf_node(0, depth=2)
    lr = create_leaf_node(1, depth=2)
    rl = create_leaf_node(0, depth=2)
    rr = create_leaf_node(1, depth=2)
    left_internal = create_internal_node(0, 0.3, ll, lr, depth=1)
    right_internal = create_internal_node(0, 0.7, rl, rr, depth=1)
    root = create_internal_node(0, 0.5, left_internal, right_internal, depth=0)
    return TreeGenotype(root=root, n_features=4, n_classes=2)


# ---------------------------------------------------------------------------
# calculate_feature_frequency
# ---------------------------------------------------------------------------


class TestCalculateFeatureFrequency:
    """Tests for FeatureImportanceAnalyzer.calculate_feature_frequency."""

    def test_returns_dict(self):
        tree = _single_split_tree()
        result = FeatureImportanceAnalyzer.calculate_feature_frequency(tree)
        assert isinstance(result, dict)

    def test_leaf_only_returns_empty(self):
        """A tree with only a leaf has no internal nodes → empty frequency dict."""
        tree = _leaf_only_tree()
        result = FeatureImportanceAnalyzer.calculate_feature_frequency(tree)
        assert result == {}

    def test_single_split_counts_one(self):
        """A depth-1 tree splitting on feature 2 should record feature 2 once."""
        tree = _single_split_tree(feature_idx=2)
        result = FeatureImportanceAnalyzer.calculate_feature_frequency(tree)
        assert result == {2: 1}

    def test_two_feature_tree_counts(self):
        """The two-feature tree uses feature 0 once (root) and feature 1 once."""
        tree = _two_feature_tree()
        result = FeatureImportanceAnalyzer.calculate_feature_frequency(tree)
        assert result.get(0) == 1, f"Feature 0 count wrong: {result}"
        assert result.get(1) == 1, f"Feature 1 count wrong: {result}"
        assert 2 not in result  # feature 2 is a leaf prediction, not a split

    def test_repeated_feature_counts_correctly(self):
        """A tree using feature 0 at three internal nodes should count it as 3."""
        tree = _repeated_feature_tree()
        result = FeatureImportanceAnalyzer.calculate_feature_frequency(tree)
        assert result.get(0) == 3, f"Expected 3, got {result}"
        # No other features should appear
        assert list(result.keys()) == [0]

    def test_values_are_positive_integers(self):
        tree = _two_feature_tree()
        result = FeatureImportanceAnalyzer.calculate_feature_frequency(tree)
        for feature, count in result.items():
            assert isinstance(count, int) and count > 0

    def test_total_count_equals_internal_nodes(self):
        """Sum of all frequencies must equal the number of internal nodes."""
        tree = _two_feature_tree()
        result = FeatureImportanceAnalyzer.calculate_feature_frequency(tree)
        n_internal = len(tree.get_internal_nodes())
        assert sum(result.values()) == n_internal


# ---------------------------------------------------------------------------
# calculate_feature_depth_importance
# ---------------------------------------------------------------------------


class TestCalculateFeatureDepthImportance:
    """Tests for FeatureImportanceAnalyzer.calculate_feature_depth_importance."""

    def test_returns_dict(self):
        tree = _single_split_tree()
        result = FeatureImportanceAnalyzer.calculate_feature_depth_importance(tree)
        assert isinstance(result, dict)

    def test_leaf_only_returns_empty(self):
        tree = _leaf_only_tree()
        result = FeatureImportanceAnalyzer.calculate_feature_depth_importance(tree)
        assert result == {}

    def test_values_sum_to_one(self):
        """Normalized importances must sum to 1.0."""
        tree = _two_feature_tree()
        result = FeatureImportanceAnalyzer.calculate_feature_depth_importance(tree)
        assert sum(result.values()) == pytest.approx(1.0, abs=1e-9)

    def test_single_split_feature_gets_full_importance(self):
        """A depth-1 tree has only one feature; it should get importance 1.0."""
        tree = _single_split_tree(feature_idx=3)
        result = FeatureImportanceAnalyzer.calculate_feature_depth_importance(tree)
        assert result.get(3) == pytest.approx(1.0, abs=1e-9)

    def test_root_feature_outweighs_deeper_same_feature(self):
        """Feature at depth 0 contributes 1/(0+1)=1.0; at depth 1 it's 1/(1+1)=0.5.
        Using two different features lets us compare root vs. child importance."""
        # root splits on feat 0 (depth 0), child splits on feat 1 (depth 1)
        tree = _two_feature_tree()
        result = FeatureImportanceAnalyzer.calculate_feature_depth_importance(tree)
        # feat 0 weight = 1.0, feat 1 weight = 0.5 → after normalization feat 0 > feat 1
        assert result[0] > result[1], (
            f"Expected feat-0 importance ({result[0]:.3f}) > feat-1 ({result[1]:.3f})"
        )

    def test_all_importances_in_range(self):
        """All importance values must be in [0, 1]."""
        tree = _repeated_feature_tree()
        result = FeatureImportanceAnalyzer.calculate_feature_depth_importance(tree)
        for feature, score in result.items():
            assert 0.0 <= score <= 1.0, f"Feature {feature} importance out of range: {score}"

    def test_repeated_feature_at_multiple_depths(self):
        """Feature 0 at depths 0, 1, 1 → weights 1.0 + 0.5 + 0.5 = 2.0 (then normalized to 1.0)."""
        tree = _repeated_feature_tree()
        result = FeatureImportanceAnalyzer.calculate_feature_depth_importance(tree)
        # Only feature 0 exists → it must get importance 1.0
        assert set(result.keys()) == {0}
        assert result[0] == pytest.approx(1.0, abs=1e-9)

    def test_keys_match_feature_frequency_keys(self):
        """Keys from depth-importance and frequency should be the same set."""
        tree = _two_feature_tree()
        freq_keys = set(FeatureImportanceAnalyzer.calculate_feature_frequency(tree).keys())
        depth_keys = set(
            FeatureImportanceAnalyzer.calculate_feature_depth_importance(tree).keys()
        )
        assert freq_keys == depth_keys

    def test_different_trees_give_different_importances(self):
        """Two structurally different trees should yield different importances."""
        tree_a = _single_split_tree(feature_idx=0)
        tree_b = _single_split_tree(feature_idx=2)
        imp_a = FeatureImportanceAnalyzer.calculate_feature_depth_importance(tree_a)
        imp_b = FeatureImportanceAnalyzer.calculate_feature_depth_importance(tree_b)
        assert imp_a != imp_b
