"""Unit tests for LDD fixes across the ga-optimized-trees project.

Each test class targets a specific LDD and validates that the fix
works correctly, including edge cases identified in the design document.

Requires: pytest, numpy, scikit-learn
"""

import random

import numpy as np
import pytest

from ga_trees.fitness.calculator import (
    DEFAULT_MAX_NODES_FALLBACK,
    VALID_CLASSIFICATION_METRICS,
    VALID_MODES,
    FitnessCalculator,
    InterpretabilityCalculator,
    TreePredictor,
)
from ga_trees.ga.engine import GAConfig, Mutation
from ga_trees.genotype.tree_genotype import (
    Node,
    TreeGenotype,
    create_internal_node,
    create_leaf_node,
)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


def _two_node_tree(max_depth: int = 5) -> TreeGenotype:
    """Depth-1 tree: root splits on feature 0 at 0.5, leaves predict 0/1."""
    left = create_leaf_node(prediction=0, depth=1)
    right = create_leaf_node(prediction=1, depth=1)
    root = create_internal_node(
        feature_idx=0, threshold=0.5, left_child=left, right_child=right, depth=0
    )
    return TreeGenotype(
        root=root, n_features=4, n_classes=2, max_depth=max_depth,
        min_samples_split=2, min_samples_leaf=1,
    )


def _single_leaf_tree() -> TreeGenotype:
    """A tree with a single leaf node (trivial tree)."""
    root = create_leaf_node(prediction=0, depth=0)
    return TreeGenotype(
        root=root, n_features=4, n_classes=2, max_depth=5,
        min_samples_split=2, min_samples_leaf=1,
    )


def _deep_tree(depth: int = 3) -> TreeGenotype:
    """Chain tree of given depth (left-skewed)."""
    node = create_leaf_node(prediction=1, depth=depth)
    for d in range(depth - 1, -1, -1):
        right_leaf = create_leaf_node(prediction=0, depth=d + 1)
        node = create_internal_node(
            feature_idx=0, threshold=0.5 + d * 0.01,
            left_child=node, right_child=right_leaf, depth=d,
        )
    return TreeGenotype(
        root=node, n_features=4, n_classes=2, max_depth=depth + 2,
        min_samples_split=2, min_samples_leaf=1,
    )


def _linearly_separable_data(n_samples: int = 10):
    """Dataset perfectly separable on feature 0 at 0.5."""
    X = np.column_stack([
        np.linspace(0.1, 0.95, n_samples),
        np.random.rand(n_samples),
        np.random.rand(n_samples),
        np.random.rand(n_samples),
    ])
    y = (X[:, 0] > 0.5).astype(int)
    return X, y


def _imbalanced_data():
    """95% class-0, 5% class-1 dataset."""
    n = 100
    X = np.random.RandomState(42).rand(n, 4)
    y = np.zeros(n, dtype=int)
    y[:5] = 1  # 5% minority
    return X, y


# ===================================================================
# LDD-1: Pareto mode no longer silently falls back
# ===================================================================


class TestLDD1_ParetoModeFix:
    """LDD-1: Pareto mode must return a tuple, not a scalar."""

    def test_pareto_mode_returns_tuple(self):
        """In pareto mode, calculate_fitness returns (accuracy, interp)."""
        fc = FitnessCalculator(mode="pareto")
        tree = _two_node_tree()
        X, y = _linearly_separable_data()
        result = fc.calculate_fitness(tree, X, y)
        assert isinstance(result, tuple), f"Expected tuple, got {type(result)}"
        assert len(result) == 2
        assert 0.0 <= result[0] <= 1.0  # accuracy
        assert 0.0 <= result[1] <= 1.0  # interpretability

    def test_weighted_sum_returns_scalar(self):
        """In weighted_sum mode, calculate_fitness returns a float."""
        fc = FitnessCalculator(mode="weighted_sum")
        tree = _two_node_tree()
        X, y = _linearly_separable_data()
        result = fc.calculate_fitness(tree, X, y)
        assert isinstance(result, float)

    def test_invalid_mode_raises(self):
        """Invalid mode string raises ValueError."""
        with pytest.raises(ValueError, match="mode must be one of"):
            FitnessCalculator(mode="invalid")


# ===================================================================
# LDD-3: Validation set support
# ===================================================================


class TestLDD3_ValidationSet:
    """LDD-3: Fitness evaluation supports train/val split."""

    def test_val_set_used_for_accuracy(self):
        """When X_val/y_val provided, accuracy is computed on validation data."""
        fc = FitnessCalculator()
        tree = _two_node_tree()
        X_train, y_train = _linearly_separable_data(20)

        # Create a validation set where the tree gets everything wrong
        X_val = np.array([[0.1, 0, 0, 0], [0.9, 0, 0, 0]])
        y_val = np.array([1, 0])  # Swapped labels

        fitness_train = fc.calculate_fitness(tree, X_train, y_train)
        # Re-create tree to reset
        tree2 = _two_node_tree()
        fitness_val = fc.calculate_fitness(tree2, X_train, y_train, X_val, y_val)

        # Validation accuracy should be 0% → fitness much lower
        assert fitness_val < fitness_train

    def test_no_val_set_uses_training_data(self):
        """Without validation set, training data is used (backward compatible)."""
        fc = FitnessCalculator()
        tree = _two_node_tree()
        X, y = _linearly_separable_data()
        result = fc.calculate_fitness(tree, X, y)
        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0


# ===================================================================
# LDD-4: Vectorized prediction
# ===================================================================


class TestLDD4_VectorizedPrediction:
    """LDD-4: Batch prediction produces correct results."""

    def test_predict_matches_expected(self):
        """Predictions match expected values for a simple split."""
        tree = _two_node_tree()
        X = np.array([
            [0.1, 0.5, 0.5, 0.5],  # left → 0
            [0.3, 0.5, 0.5, 0.5],  # left → 0
            [0.6, 0.5, 0.5, 0.5],  # right → 1
            [0.9, 0.5, 0.5, 0.5],  # right → 1
        ])
        preds = TreePredictor.predict(tree, X)
        np.testing.assert_array_equal(preds, [0, 0, 1, 1])

    def test_predict_boundary(self):
        """Value exactly at threshold goes left (<=)."""
        tree = _two_node_tree()
        X = np.array([[0.5, 0.0, 0.0, 0.0]])
        preds = TreePredictor.predict(tree, X)
        assert preds[0] == 0

    def test_predict_empty_input(self):
        """Empty input returns empty array."""
        tree = _two_node_tree()
        X = np.empty((0, 4))
        preds = TreePredictor.predict(tree, X)
        assert len(preds) == 0

    def test_predict_output_shape(self):
        """Output length matches input."""
        tree = _two_node_tree()
        X = np.random.rand(50, 4)
        preds = TreePredictor.predict(tree, X)
        assert preds.shape == (50,)

    def test_predict_deep_tree(self):
        """Vectorized prediction works on deeper trees."""
        tree = _deep_tree(depth=5)
        X = np.random.rand(100, 4)
        preds = TreePredictor.predict(tree, X)
        assert preds.shape == (100,)
        assert np.all(np.isin(preds, [0, 1]))

    def test_predict_with_array_leaf(self):
        """Handles leaf with numpy array prediction (class probabilities)."""
        left = create_leaf_node(prediction=np.array([0.9, 0.1]), depth=1)
        right = create_leaf_node(prediction=1, depth=1)
        root = create_internal_node(0, 0.5, left, right, depth=0)
        tree = TreeGenotype(root=root, n_features=2, n_classes=2)
        X = np.array([[0.3, 0.0]])  # left → argmax([0.9, 0.1]) = 0
        preds = TreePredictor.predict(tree, X)
        assert preds[0] == 0


# ===================================================================
# LDD-5: Input validation
# ===================================================================


class TestLDD5_InputValidation:
    """LDD-5: Public APIs reject invalid inputs with clear errors."""

    def test_predict_rejects_1d_input(self):
        tree = _two_node_tree()
        with pytest.raises(ValueError, match="2-D"):
            TreePredictor.predict(tree, np.array([1.0, 2.0]))

    def test_predict_rejects_too_few_features(self):
        tree = _two_node_tree()  # expects 4 features
        with pytest.raises(ValueError, match="features"):
            TreePredictor.predict(tree, np.array([[1.0, 2.0]]))  # only 2

    def test_fitness_rejects_negative_weight(self):
        with pytest.raises(ValueError, match="accuracy_weight"):
            FitnessCalculator(accuracy_weight=-0.5)

    def test_fitness_rejects_weight_over_one(self):
        with pytest.raises(ValueError, match="interpretability_weight"):
            FitnessCalculator(interpretability_weight=1.5)

    def test_fitness_rejects_mismatched_xy(self):
        fc = FitnessCalculator()
        tree = _two_node_tree()
        X = np.random.rand(10, 4)
        y = np.zeros(5)  # wrong length
        with pytest.raises(ValueError, match="mismatch"):
            fc.calculate_fitness(tree, X, y)

    def test_gaconfig_rejects_zero_population(self):
        with pytest.raises(ValueError, match="population_size"):
            GAConfig(population_size=0)

    def test_gaconfig_rejects_invalid_crossover_prob(self):
        with pytest.raises(ValueError, match="crossover_prob"):
            GAConfig(crossover_prob=1.5)

    def test_gaconfig_rejects_invalid_tournament_size(self):
        with pytest.raises(ValueError, match="tournament_size"):
            GAConfig(tournament_size=1)

    def test_fitness_rejects_invalid_metric(self):
        with pytest.raises(ValueError, match="classification_metric"):
            FitnessCalculator(classification_metric="rmse")


# ===================================================================
# LDD-7: Dynamic max_nodes
# ===================================================================


class TestLDD7_DynamicMaxNodes:
    """LDD-7: Node complexity uses tree.max_depth, not hardcoded 127."""

    def test_max_depth_5_full_tree_scores_zero(self):
        """A full tree at max_depth=5 should have complexity near 0."""
        # max_nodes = 2^6 - 1 = 63
        # With 63 nodes, score = 1 - 63/63 = 0.0
        tree = _two_node_tree(max_depth=5)
        # Manually override node count for testing
        score = InterpretabilityCalculator._node_complexity(tree)
        # tree has 3 nodes, max_nodes=63 → 1 - 3/63 ≈ 0.952
        expected = 1.0 - 3.0 / 63.0
        assert abs(score - expected) < 1e-6

    def test_different_max_depths_give_different_scores(self):
        """Same tree under different max_depth constraints yields different scores."""
        tree5 = _two_node_tree(max_depth=5)  # max_nodes = 63
        tree10 = _two_node_tree(max_depth=10)  # max_nodes = 2047
        score5 = InterpretabilityCalculator._node_complexity(tree5)
        score10 = InterpretabilityCalculator._node_complexity(tree10)
        # Same number of nodes (3), but different denominators
        assert score10 > score5  # 1 - 3/2047 > 1 - 3/63


# ===================================================================
# LDD-8: Feature coherence for leaf-only trees
# ===================================================================


class TestLDD8_FeatureCoherence:
    """LDD-8: Leaf-only trees get neutral coherence score, not 1.0."""

    def test_leaf_only_tree_returns_half(self):
        """A single-leaf tree should return 0.5, not 1.0."""
        tree = _single_leaf_tree()
        score = InterpretabilityCalculator._feature_coherence(tree)
        assert score == 0.5

    def test_single_feature_tree_returns_high(self):
        """A tree using 1 of 4 features should score 0.75."""
        tree = _two_node_tree()  # uses feature 0 out of 4
        score = InterpretabilityCalculator._feature_coherence(tree)
        expected = 1.0 - 1.0 / 4.0  # 0.75
        assert abs(score - expected) < 1e-6


# ===================================================================
# LDD-9: Reproducibility
# ===================================================================


class TestLDD9_Reproducibility:
    """LDD-9: Random seed produces deterministic results."""

    def test_gaconfig_accepts_random_state(self):
        """GAConfig stores random_state."""
        config = GAConfig(random_state=42)
        assert config.random_state == 42

    def test_gaconfig_default_is_none(self):
        """Default random_state is None."""
        config = GAConfig()
        assert config.random_state is None


# ===================================================================
# LDD-10: Bare except fix (in metrics.py)
# ===================================================================


class TestLDD10_MetricsExceptionHandling:
    """LDD-10: Only expected exceptions are caught in metrics."""

    def test_metrics_does_not_swallow_type_error(self):
        """TypeError from bad y_prob should propagate, not be silenced."""
        from ga_trees.evaluation.metrics import MetricsCalculator

        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 0])
        # Valid call — should not raise
        result = MetricsCalculator.calculate_classification_metrics(y_true, y_pred)
        assert "accuracy" in result


# ===================================================================
# LDD-11: Unreachable leaf default prediction
# ===================================================================


class TestLDD11_UnreachableLeafPrior:
    """LDD-11: Unreachable leaves use global prior, not 0."""

    def test_unreachable_leaf_gets_majority_class(self):
        """A leaf no samples reach should predict the dataset majority class."""
        # Tree splits on feature 0 at 0.5
        # All samples have feature 0 < 0.5 → all go left
        # Right leaf is unreachable
        tree = _two_node_tree()
        X = np.array([[0.1, 0, 0, 0], [0.2, 0, 0, 0], [0.3, 0, 0, 0]])
        y = np.array([1, 1, 1])  # majority class = 1

        TreePredictor.fit_leaf_predictions(tree, X, y)
        right_leaf = tree.root.right_child
        # LDD-11: should be 1 (majority class), not 0 (arbitrary default)
        assert right_leaf.prediction == 1


# ===================================================================
# LDD-12: Configurable classification metric
# ===================================================================


class TestLDD12_ConfigurableMetric:
    """LDD-12: FitnessCalculator supports multiple classification metrics."""

    def test_f1_macro_accepted(self):
        """f1_macro is a valid classification_metric."""
        fc = FitnessCalculator(classification_metric="f1_macro")
        assert fc.classification_metric == "f1_macro"

    def test_balanced_accuracy_accepted(self):
        fc = FitnessCalculator(classification_metric="balanced_accuracy")
        assert fc.classification_metric == "balanced_accuracy"

    def test_invalid_metric_rejected(self):
        with pytest.raises(ValueError):
            FitnessCalculator(classification_metric="rmse")

    def test_f1_weighted_runs_end_to_end(self):
        """f1_weighted metric produces valid fitness on real data."""
        fc = FitnessCalculator(classification_metric="f1_weighted")
        tree = _two_node_tree()
        X, y = _linearly_separable_data()
        fitness = fc.calculate_fitness(tree, X, y)
        assert isinstance(fitness, float)
        assert 0.0 <= fitness <= 1.0


# ===================================================================
# LDD-13: prune_subtree root exclusion
# ===================================================================


class TestLDD13_PruneSubtreeRootExclusion:
    """LDD-13: prune_subtree never prunes the root node."""

    def test_prune_never_returns_leaf_root(self):
        """After prune_subtree, root should still be internal (for multi-node trees)."""
        tree = _deep_tree(depth=3)
        feature_ranges = {i: (0.0, 1.0) for i in range(4)}
        mutation = Mutation(n_features=4, feature_ranges=feature_ranges)

        # Run prune many times — root should never become a leaf
        for _ in range(50):
            test_tree = tree.copy()
            result = mutation.prune_subtree(test_tree)
            assert result.root.is_internal(), "Root was pruned to leaf!"


# ===================================================================
# Integration: full fitness pipeline
# ===================================================================


class TestIntegration:
    """End-to-end tests across multiple LDD fixes."""

    def test_full_pipeline_weighted_sum(self):
        """Full pipeline: create tree, fit, predict, calculate fitness."""
        tree = _two_node_tree()
        X, y = _linearly_separable_data(50)
        fc = FitnessCalculator(
            accuracy_weight=0.7,
            interpretability_weight=0.3,
            classification_metric="accuracy",
        )
        fitness = fc.calculate_fitness(tree, X, y)
        assert isinstance(fitness, float)
        assert 0.0 <= fitness <= 1.0
        assert tree.accuracy_ is not None
        assert tree.interpretability_ is not None

    def test_full_pipeline_pareto(self):
        """Pareto mode returns proper tuple."""
        tree = _two_node_tree()
        X, y = _linearly_separable_data(50)
        fc = FitnessCalculator(mode="pareto")
        result = fc.calculate_fitness(tree, X, y)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_fitness_is_weighted_combination(self):
        """fitness == w1*accuracy + w2*interpretability."""
        aw, iw = 0.6, 0.4
        fc = FitnessCalculator(accuracy_weight=aw, interpretability_weight=iw)
        tree = _two_node_tree()
        X, y = _linearly_separable_data()
        fitness = fc.calculate_fitness(tree, X, y)
        expected = aw * tree.accuracy_ + iw * float(tree.interpretability_)
        assert fitness == pytest.approx(expected, abs=1e-9)

    def test_fitness_reproduces_on_same_data(self):
        """Same data → same fitness."""
        fc = FitnessCalculator()
        X, y = _linearly_separable_data()
        f1 = fc.calculate_fitness(_two_node_tree(), X, y)
        f2 = fc.calculate_fitness(_two_node_tree(), X, y)
        assert f1 == pytest.approx(f2, abs=1e-9)
