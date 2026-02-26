"""Unit tests for src/ga_trees/fitness/calculator.py.

Covers:
- FitnessCalculator (weighted_sum mode)
- TreePredictor.predict and fit_leaf_predictions
- InterpretabilityCalculator composite score and sub-metrics
"""

import numpy as np
import pytest

from ga_trees.fitness.calculator import FitnessCalculator, InterpretabilityCalculator, TreePredictor
from ga_trees.genotype.tree_genotype import TreeGenotype, create_internal_node, create_leaf_node

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _two_node_tree():
    """Depth-1 tree: root splits on feature 0 at 0.5, leaves predict 0 / 1."""
    left = create_leaf_node(prediction=0, depth=1)
    right = create_leaf_node(prediction=1, depth=1)
    root = create_internal_node(
        feature_idx=0, threshold=0.5, left_child=left, right_child=right, depth=0
    )
    return TreeGenotype(
        root=root,
        n_features=4,
        n_classes=2,
        max_depth=5,
        min_samples_split=2,
        min_samples_leaf=1,
    )


def _three_level_tree():
    """Depth-2 tree with 7 nodes, using features 0 and 1."""
    ll = create_leaf_node(0, depth=2)
    lr = create_leaf_node(1, depth=2)
    rl = create_leaf_node(0, depth=2)
    rr = create_leaf_node(1, depth=2)
    left_internal = create_internal_node(1, 0.3, ll, lr, depth=1)
    right_internal = create_internal_node(1, 0.7, rl, rr, depth=1)
    root = create_internal_node(0, 0.5, left_internal, right_internal, depth=0)
    return TreeGenotype(root=root, n_features=4, n_classes=2, max_depth=5)


def _linearly_separable_data():
    """10-sample dataset perfectly separable on feature 0 at 0.5."""
    X = np.array(
        [
            [0.1, 0.9, 0.5, 0.5],
            [0.2, 0.8, 0.5, 0.5],
            [0.3, 0.7, 0.5, 0.5],
            [0.4, 0.6, 0.5, 0.5],
            [0.45, 0.55, 0.5, 0.5],
            [0.6, 0.4, 0.5, 0.5],
            [0.7, 0.3, 0.5, 0.5],
            [0.8, 0.2, 0.5, 0.5],
            [0.9, 0.1, 0.5, 0.5],
            [0.95, 0.05, 0.5, 0.5],
        ]
    )
    y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    return X, y


# ---------------------------------------------------------------------------
# TreePredictor tests
# ---------------------------------------------------------------------------


class TestTreePredictor:
    """Tests for TreePredictor static methods."""

    def test_predict_routes_left(self):
        """Samples with feature-0 <= 0.5 should route to the left leaf (class 0)."""
        tree = _two_node_tree()
        X = np.array([[0.1, 0.5, 0.5, 0.5], [0.4, 0.5, 0.5, 0.5]])
        preds = TreePredictor.predict(tree, X)
        assert np.all(preds == 0), f"Expected all 0, got {preds}"

    def test_predict_routes_right(self):
        """Samples with feature-0 > 0.5 should route to the right leaf (class 1)."""
        tree = _two_node_tree()
        X = np.array([[0.6, 0.5, 0.5, 0.5], [0.9, 0.5, 0.5, 0.5]])
        preds = TreePredictor.predict(tree, X)
        assert np.all(preds == 1), f"Expected all 1, got {preds}"

    def test_predict_output_shape(self):
        """Output array length must match number of input samples."""
        tree = _two_node_tree()
        X = np.random.rand(20, 4)
        preds = TreePredictor.predict(tree, X)
        assert preds.shape == (20,)

    def test_predict_boundary_exact(self):
        """Sample exactly at threshold (feature-0 == 0.5) routes to the left leaf."""
        tree = _two_node_tree()
        X = np.array([[0.5, 0.0, 0.0, 0.0]])
        preds = TreePredictor.predict(tree, X)
        assert preds[0] == 0  # <= threshold → left → class 0

    def test_predict_deeper_tree(self):
        """predict works correctly on a depth-2 tree."""
        tree = _three_level_tree()
        # feature 0 < 0.5 → left subtree, feature 1 < 0.3 → ll (class 0)
        X = np.array([[0.2, 0.1, 0.0, 0.0]])
        preds = TreePredictor.predict(tree, X)
        assert preds[0] == 0

    def test_fit_leaf_predictions_majority_vote(self):
        """fit_leaf_predictions updates leaves via majority-vote for classification."""
        tree = _two_node_tree()
        X, y = _linearly_separable_data()
        TreePredictor.fit_leaf_predictions(tree, X, y)

        left_leaf = tree.root.left_child
        right_leaf = tree.root.right_child
        # All class-0 samples go left, all class-1 go right
        assert left_leaf.prediction == 0
        assert right_leaf.prediction == 1

    def test_fit_leaf_predictions_updates_all_reached_leaves(self):
        """Every leaf that receives at least one sample gets an updated prediction."""
        tree = _two_node_tree()
        X, y = _linearly_separable_data()
        TreePredictor.fit_leaf_predictions(tree, X, y)
        # Both leaves should be reachable with this balanced dataset
        assert tree.root.left_child.prediction is not None
        assert tree.root.right_child.prediction is not None

    def test_fit_then_predict_perfect_accuracy(self):
        """fit_leaf_predictions + predict achieves 100% on linearly separable data."""
        tree = _two_node_tree()
        X, y = _linearly_separable_data()
        TreePredictor.fit_leaf_predictions(tree, X, y)
        preds = TreePredictor.predict(tree, X)
        accuracy = np.mean(preds == y)
        assert accuracy == 1.0, f"Expected 1.0 accuracy, got {accuracy}"

    def test_predict_with_array_prediction_leaf(self):
        """predict handles a leaf whose prediction attribute is a numpy array (class probs)."""
        left = create_leaf_node(prediction=np.array([0.9, 0.1]), depth=1)
        right = create_leaf_node(prediction=1, depth=1)
        root = create_internal_node(0, 0.5, left, right, depth=0)
        tree = TreeGenotype(root=root, n_features=2, n_classes=2)

        X = np.array([[0.3, 0.0]])  # routes left → argmax([0.9, 0.1]) = 0
        preds = TreePredictor.predict(tree, X)
        assert preds[0] == 0


# ---------------------------------------------------------------------------
# InterpretabilityCalculator tests
# ---------------------------------------------------------------------------


class TestInterpretabilityCalculator:
    """Tests for InterpretabilityCalculator."""

    @pytest.fixture
    def simple_tree(self):
        return _two_node_tree()

    @pytest.fixture
    def deep_tree(self):
        return _three_level_tree()

    @pytest.fixture
    def default_weights(self):
        return {
            "node_complexity": 0.4,
            "feature_coherence": 0.3,
            "tree_balance": 0.2,
            "semantic_coherence": 0.1,
        }

    def test_composite_score_range(self, simple_tree, default_weights):
        """Composite score must be in [0, 1]."""
        score = InterpretabilityCalculator.calculate_composite_score(simple_tree, default_weights)
        assert 0.0 <= score <= 1.0, f"Score out of range: {score}"

    def test_composite_score_is_float(self, simple_tree, default_weights):
        """Composite score must be a Python or numpy scalar float."""
        score = InterpretabilityCalculator.calculate_composite_score(simple_tree, default_weights)
        assert isinstance(float(score), float)

    def test_small_tree_higher_complexity_score(self, simple_tree, deep_tree, default_weights):
        """A smaller tree should have a higher node-complexity contribution."""
        weights = {"node_complexity": 1.0}
        small_score = InterpretabilityCalculator.calculate_composite_score(simple_tree, weights)
        deep_score = InterpretabilityCalculator.calculate_composite_score(deep_tree, weights)
        assert (
            small_score > deep_score
        ), f"Smaller tree ({small_score:.3f}) should beat deeper tree ({deep_score:.3f})"

    def test_partial_weights_subset(self, simple_tree):
        """Passing only a subset of weight keys should not raise."""
        score = InterpretabilityCalculator.calculate_composite_score(
            simple_tree, {"node_complexity": 1.0}
        )
        assert 0.0 <= score <= 1.0

    def test_empty_weights_returns_zero(self, simple_tree):
        """Empty weight dict → score is 0.0."""
        score = InterpretabilityCalculator.calculate_composite_score(simple_tree, {})
        assert score == 0.0

    def test_balanced_tree_gets_full_balance_score(self):
        """A perfectly balanced depth-1 tree should score 1.0 on tree_balance alone."""
        tree = _two_node_tree()
        score = InterpretabilityCalculator.calculate_composite_score(tree, {"tree_balance": 1.0})
        assert score == pytest.approx(1.0, abs=1e-6)

    def test_semantic_coherence_uniform_predictions(self):
        """All leaves predicting the same class → semantic coherence = 1.0."""
        left = create_leaf_node(0, depth=1)
        right = create_leaf_node(0, depth=1)  # same class as left
        root = create_internal_node(0, 0.5, left, right, depth=0)
        tree = TreeGenotype(root=root, n_features=2, n_classes=2)

        score = InterpretabilityCalculator.calculate_composite_score(
            tree, {"semantic_coherence": 1.0}
        )
        assert score == pytest.approx(1.0, abs=1e-6)


# ---------------------------------------------------------------------------
# FitnessCalculator tests
# ---------------------------------------------------------------------------


class TestFitnessCalculator:
    """Tests for FitnessCalculator in weighted_sum mode."""

    @pytest.fixture
    def simple_tree(self):
        return _two_node_tree()

    @pytest.fixture
    def separable_data(self):
        return _linearly_separable_data()

    def test_default_mode_is_weighted_sum(self):
        fc = FitnessCalculator()
        assert fc.mode == "weighted_sum"

    def test_weights_sum_to_one(self):
        fc = FitnessCalculator(accuracy_weight=0.7, interpretability_weight=0.3)
        assert fc.accuracy_weight + fc.interpretability_weight == pytest.approx(1.0)

    def test_fitness_is_scalar_in_range(self, simple_tree, separable_data):
        X, y = separable_data
        fc = FitnessCalculator()
        fitness = fc.calculate_fitness(simple_tree, X, y)
        assert isinstance(float(fitness), float)
        assert 0.0 <= fitness <= 1.0, f"Fitness out of [0,1]: {fitness}"

    def test_fitness_sets_tree_accuracy(self, simple_tree, separable_data):
        X, y = separable_data
        fc = FitnessCalculator()
        fc.calculate_fitness(simple_tree, X, y)
        assert hasattr(simple_tree, "accuracy_")
        assert 0.0 <= simple_tree.accuracy_ <= 1.0

    def test_fitness_sets_tree_interpretability(self, simple_tree, separable_data):
        X, y = separable_data
        fc = FitnessCalculator()
        fc.calculate_fitness(simple_tree, X, y)
        assert hasattr(simple_tree, "interpretability_")
        assert 0.0 <= float(simple_tree.interpretability_) <= 1.0

    def test_perfect_tree_high_fitness(self, separable_data):
        """A tree that perfectly separates the data should have high fitness."""
        X, y = separable_data
        fc = FitnessCalculator(accuracy_weight=0.8, interpretability_weight=0.2)
        tree = _two_node_tree()
        fitness = fc.calculate_fitness(tree, X, y)
        assert fitness > 0.7, f"Expected high fitness for perfect tree, got {fitness}"

    def test_custom_accuracy_weight(self, simple_tree, separable_data):
        """Higher accuracy weight should dominate fitness when accuracy is high."""
        X, y = separable_data
        fc_high = FitnessCalculator(accuracy_weight=0.99, interpretability_weight=0.01)
        fc_low = FitnessCalculator(accuracy_weight=0.01, interpretability_weight=0.99)
        fit_high = fc_high.calculate_fitness(_two_node_tree(), X, y)
        fit_low = fc_low.calculate_fitness(_two_node_tree(), X, y)
        # With perfect accuracy (1.0) and interpretability < 1.0,
        # the high-accuracy-weight version should yield higher fitness
        assert fit_high > fit_low

    def test_fitness_is_weighted_combination(self, separable_data):
        """Fitness ≈ accuracy_weight * accuracy + interpretability_weight * interp."""
        X, y = separable_data
        aw, iw = 0.6, 0.4
        fc = FitnessCalculator(accuracy_weight=aw, interpretability_weight=iw)
        tree = _two_node_tree()
        fitness = fc.calculate_fitness(tree, X, y)
        expected = aw * tree.accuracy_ + iw * float(tree.interpretability_)
        assert fitness == pytest.approx(expected, abs=1e-9)

    def test_fitness_reproduces_on_same_data(self, separable_data):
        """Calling calculate_fitness twice with same data returns same result."""
        X, y = separable_data
        fc = FitnessCalculator()
        tree = _two_node_tree()
        f1 = fc.calculate_fitness(tree, X, y)
        f2 = fc.calculate_fitness(tree, X, y)
        assert f1 == pytest.approx(f2, abs=1e-9)

    def test_custom_interpretability_weights(self, separable_data):
        """Custom interpretability sub-weights are forwarded and applied."""
        X, y = separable_data
        fc = FitnessCalculator(
            interpretability_weights={
                "node_complexity": 1.0,
                "feature_coherence": 0.0,
                "tree_balance": 0.0,
                "semantic_coherence": 0.0,
            }
        )
        tree = _two_node_tree()
        fitness = fc.calculate_fitness(tree, X, y)
        assert isinstance(float(fitness), float)
        assert 0.0 <= fitness <= 1.0

    @pytest.mark.slow
    def test_fitness_on_iris_dataset(self):
        """Fitness calculator works end-to-end on a real sklearn dataset."""
        from sklearn.datasets import load_iris

        X, y = load_iris(return_X_y=True)
        fc = FitnessCalculator()
        tree = _two_node_tree()
        fitness = fc.calculate_fitness(tree, X[:, :4], y)
        assert 0.0 <= fitness <= 1.0
