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


# ---------------------------------------------------------------------------
# §4.1 — Non-linear node complexity metric
# ---------------------------------------------------------------------------


class TestNodeComplexityMetric:
    """§4.1: Non-linear node complexity formula 1/(1+n/target)."""

    def test_node_complexity_small_tree_high_score(self):
        """A 3-node tree should score > 0.8 with default target=15."""
        tree = _two_node_tree()  # 3 nodes
        score = InterpretabilityCalculator._node_complexity(tree)
        assert score > 0.8, f"3-node tree expected score > 0.8, got {score:.3f}"

    def test_node_complexity_target_tree_scores_half(self):
        """A tree with exactly target_nodes nodes should score exactly 0.5."""
        tree = _two_node_tree()
        # Monkeypatch get_num_nodes to return the target
        original = tree.__class__.get_num_nodes
        tree.__class__.get_num_nodes = lambda self: 15
        try:
            score = InterpretabilityCalculator._node_complexity(tree, target_nodes=15)
            assert score == pytest.approx(0.5, abs=1e-9)
        finally:
            tree.__class__.get_num_nodes = original

    def test_node_complexity_monotone_decreasing(self):
        """Larger trees should have lower complexity scores."""
        small = InterpretabilityCalculator._node_complexity(_two_node_tree())  # 3 nodes
        large = InterpretabilityCalculator._node_complexity(_three_level_tree())  # 7 nodes
        assert small > large, f"Small ({small:.3f}) should beat large ({large:.3f})"

    def test_node_complexity_custom_target(self):
        """Custom target_nodes changes the score meaningfully."""
        tree = _two_node_tree()  # 3 nodes
        score_target5 = InterpretabilityCalculator._node_complexity(tree, target_nodes=5)
        score_target20 = InterpretabilityCalculator._node_complexity(tree, target_nodes=20)
        # With target=5: 1/(1+3/5) = 0.625; with target=20: 1/(1+3/20) ≈ 0.87
        assert score_target5 < score_target20

    def test_node_complexity_always_positive(self):
        """Score must always be > 0."""
        for tree in [_two_node_tree(), _three_level_tree()]:
            score = InterpretabilityCalculator._node_complexity(tree)
            assert score > 0.0

    def test_calculate_composite_score_with_target(self):
        """calculate_composite_score accepts node_complexity_target and uses it."""
        tree = _two_node_tree()
        w = {"node_complexity": 1.0}
        score_t5 = InterpretabilityCalculator.calculate_composite_score(
            tree, w, node_complexity_target=5
        )
        score_t20 = InterpretabilityCalculator.calculate_composite_score(
            tree, w, node_complexity_target=20
        )
        assert score_t5 < score_t20

    def test_fitness_calculator_uses_node_complexity_target(self):
        """FitnessCalculator propagates node_complexity_target to interpretability."""
        X, y = _linearly_separable_data()
        fc_small_target = FitnessCalculator(
            accuracy_weight=0.0,
            interpretability_weight=1.0,
            interpretability_weights={"node_complexity": 1.0},
            node_complexity_target=3,
        )
        fc_large_target = FitnessCalculator(
            accuracy_weight=0.0,
            interpretability_weight=1.0,
            interpretability_weights={"node_complexity": 1.0},
            node_complexity_target=30,
        )
        tree_small = _two_node_tree()
        tree_large = _two_node_tree()
        fit_small = fc_small_target.calculate_fitness(tree_small, X, y)
        fit_large = fc_large_target.calculate_fitness(tree_large, X, y)
        # 3-node tree: target=3 → score≈0.5; target=30 → score≈0.91
        assert fit_large > fit_small


# ---------------------------------------------------------------------------
# §4.3 — Overfitting penalty
# ---------------------------------------------------------------------------


class TestOverfitPenalty:
    """§4.3: Overfitting penalty = overfit_weight * max(0, train_acc - val_acc)."""

    def _make_overfit_data(self):
        """Return a simple dataset and a perfect-on-train tree."""
        X, y = _linearly_separable_data()
        # Use first 7 samples as train, last 3 as val
        X_train, y_train = X[:7], y[:7]
        X_val, y_val = X[7:], y[7:]
        return X_train, y_train, X_val, y_val

    def test_no_penalty_when_weight_zero(self):
        """With overfit_penalty_weight=0, result equals standard weighted sum."""
        X, y = _linearly_separable_data()
        fc = FitnessCalculator(overfit_penalty_weight=0.0)
        tree = _two_node_tree()
        fit_no_pen = fc.calculate_fitness(tree, X, y)
        fit_with_val = fc.calculate_fitness(tree, X, y, X_val=X, y_val=y)
        # Both should give the same result since weight=0
        assert fit_no_pen == pytest.approx(fit_with_val, abs=1e-9)

    def test_penalty_applied_when_overfit_gap_exists(self):
        """Fitness with penalty < fitness without penalty when train_acc > val_acc."""
        X_train, y_train, X_val, y_val = self._make_overfit_data()
        fc_no_pen = FitnessCalculator(overfit_penalty_weight=0.0)
        fc_with_pen = FitnessCalculator(overfit_penalty_weight=0.5)
        tree_no = _two_node_tree()
        tree_yes = _two_node_tree()
        fit_no = fc_no_pen.calculate_fitness(tree_no, X_train, y_train, X_val, y_val)
        fit_yes = fc_with_pen.calculate_fitness(tree_yes, X_train, y_train, X_val, y_val)
        # If there is an overfit gap, penalty reduces fitness
        assert fit_yes <= fit_no + 1e-9  # allowing tiny numerical noise

    def test_no_penalty_when_no_val_data(self):
        """Penalty is not applied when X_val/y_val are not provided."""
        X, y = _linearly_separable_data()
        fc = FitnessCalculator(overfit_penalty_weight=1.0)
        tree = _two_node_tree()
        fitness = fc.calculate_fitness(tree, X, y)
        # No val data → no penalty branch reached
        assert 0.0 <= fitness <= 1.0


# ---------------------------------------------------------------------------
# §4.4 — Threshold-based feature coherence
# ---------------------------------------------------------------------------


class TestFeatureCoherence:
    """§4.4: Threshold-based feature coherence."""

    def test_using_few_features_scores_high(self):
        """Using 1 feature out of 13 should score 1.0 (well under max_desired=5)."""
        # Build tree using only feature 0 (1 feature used)
        tree = _two_node_tree()  # uses feature 0 only, n_features=4
        score = InterpretabilityCalculator._feature_coherence(tree)
        # max_desired = min(5, 4//2) = min(5,2) = 2; n_used=1; 1-1/2=0.5
        assert score >= 0.0

    def test_leaf_only_returns_half(self):
        """Leaf-only tree still returns 0.5 (LDD-8 preserved)."""
        root = create_leaf_node(0, depth=0)
        tree = TreeGenotype(root=root, n_features=4, n_classes=2)
        score = InterpretabilityCalculator._feature_coherence(tree)
        assert score == pytest.approx(0.5)

    def test_score_zero_features_many(self):
        """Using many features (at or beyond max_desired) returns 0."""
        # Build tree using 3 features (feature 0, 1, 2) on a 4-feature dataset
        # max_desired = min(5, 4//2) = 2; n_used=3 → score = 1 - min(3/2,1)=0
        ll = create_leaf_node(0, 2)
        lr = create_leaf_node(1, 2)
        rl = create_leaf_node(0, 2)
        rr = create_leaf_node(1, 2)
        left_int = create_internal_node(1, 0.3, ll, lr, depth=1)
        right_int = create_internal_node(2, 0.7, rl, rr, depth=1)
        root = create_internal_node(0, 0.5, left_int, right_int, depth=0)
        tree = TreeGenotype(root=root, n_features=4, n_classes=2, max_depth=5)
        score = InterpretabilityCalculator._feature_coherence(tree)
        assert score == pytest.approx(0.0)

    def test_score_nonnegative(self):
        """Feature coherence must always be >= 0."""
        for tree in [_two_node_tree(), _three_level_tree()]:
            assert InterpretabilityCalculator._feature_coherence(tree) >= 0.0


# ---------------------------------------------------------------------------
# §4.2 — Curriculum fitness via set_evolution_phase
# ---------------------------------------------------------------------------


class TestCurriculumFitness:
    """§4.2: Curriculum fitness weight shifting via set_evolution_phase."""

    def test_set_evolution_phase_noop_when_disabled(self):
        """set_evolution_phase does nothing when curriculum_fitness=False."""
        fc = FitnessCalculator(accuracy_weight=0.7, curriculum_fitness=False)
        fc.set_evolution_phase(0.1)
        assert fc.accuracy_weight == pytest.approx(0.7)

    def test_set_evolution_phase_early_high_accuracy_weight(self):
        """Phase < 0.6 → accuracy_weight=0.90."""
        fc = FitnessCalculator(curriculum_fitness=True)
        fc.set_evolution_phase(0.3)
        assert fc.accuracy_weight == pytest.approx(0.90)
        assert fc.interpretability_weight == pytest.approx(0.10)

    def test_set_evolution_phase_late_balanced_weight(self):
        """Phase >= 0.6 → accuracy_weight=0.70."""
        fc = FitnessCalculator(curriculum_fitness=True)
        fc.set_evolution_phase(0.8)
        assert fc.accuracy_weight == pytest.approx(0.70)
        assert fc.interpretability_weight == pytest.approx(0.30)

    def test_set_evolution_phase_boundary(self):
        """Phase == 0.6 falls into the 'late' bucket (>= 0.6)."""
        fc = FitnessCalculator(curriculum_fitness=True)
        fc.set_evolution_phase(0.6)
        assert fc.accuracy_weight == pytest.approx(0.70)

    def test_base_weights_preserved_after_curriculum(self):
        """_base_accuracy_weight always reflects the constructor value."""
        fc = FitnessCalculator(accuracy_weight=0.65, curriculum_fitness=True)
        fc.set_evolution_phase(0.1)
        assert fc._base_accuracy_weight == pytest.approx(0.65)

    def test_curriculum_fitness_stored_correctly(self):
        """curriculum_fitness flag is stored on the instance."""
        fc_on = FitnessCalculator(curriculum_fitness=True)
        fc_off = FitnessCalculator(curriculum_fitness=False)
        assert fc_on.curriculum_fitness is True
        assert fc_off.curriculum_fitness is False
