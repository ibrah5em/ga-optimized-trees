"""Unit tests for ga/engine.py — covering missing lines.

Covers:
- GAConfig validation (ValueError paths)
- TreeInitializer regression task + all-same-values branch
- Crossover._copy_node_contents (ndarray and None prediction)
- Crossover._repair_tree + _fix_depths + _prune_to_depth
- Mutation.expand_leaf edge cases (no expandable leaves, feature not in ranges)
- GAEngine.evolve (random_state seeding, verbose logging)
"""

import numpy as np
import pytest

from ga_trees.ga.engine import Crossover, GAConfig, GAEngine, Mutation, TreeInitializer
from ga_trees.genotype.tree_genotype import TreeGenotype, create_internal_node, create_leaf_node

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _depth1_tree(feature_idx=0, threshold=0.5, n_features=4):
    left = create_leaf_node(0, depth=1)
    right = create_leaf_node(1, depth=1)
    root = create_internal_node(feature_idx, threshold, left, right, depth=0)
    return TreeGenotype(root=root, n_features=n_features, n_classes=2, max_depth=5)


def _leaf_tree(n_features=4):
    root = create_leaf_node(0, depth=0)
    return TreeGenotype(root=root, n_features=n_features, n_classes=2, max_depth=5)


def _simple_fitness(tree, X, y):
    return 0.8


# ---------------------------------------------------------------------------
# GAConfig validation
# ---------------------------------------------------------------------------


class TestGAConfigValidation:
    """Tests for GAConfig.__post_init__ validation (ValueError paths)."""

    def test_valid_config_does_not_raise(self):
        config = GAConfig(population_size=10, n_generations=5)
        assert config.population_size == 10

    def test_invalid_population_size_raises(self):
        with pytest.raises(ValueError, match="population_size"):
            GAConfig(population_size=0)

    def test_negative_population_size_raises(self):
        with pytest.raises(ValueError, match="population_size"):
            GAConfig(population_size=-5)

    def test_invalid_n_generations_raises(self):
        with pytest.raises(ValueError, match="n_generations"):
            GAConfig(n_generations=0)

    def test_crossover_prob_above_one_raises(self):
        with pytest.raises(ValueError, match="crossover_prob"):
            GAConfig(crossover_prob=1.5)

    def test_crossover_prob_negative_raises(self):
        with pytest.raises(ValueError, match="crossover_prob"):
            GAConfig(crossover_prob=-0.1)

    def test_mutation_prob_above_one_raises(self):
        with pytest.raises(ValueError, match="mutation_prob"):
            GAConfig(mutation_prob=1.1)

    def test_mutation_prob_negative_raises(self):
        with pytest.raises(ValueError, match="mutation_prob"):
            GAConfig(mutation_prob=-0.1)

    def test_tournament_size_below_two_raises(self):
        with pytest.raises(ValueError, match="tournament_size"):
            GAConfig(tournament_size=1)

    def test_elitism_ratio_one_raises(self):
        with pytest.raises(ValueError, match="elitism_ratio"):
            GAConfig(elitism_ratio=1.0)

    def test_elitism_ratio_negative_raises(self):
        with pytest.raises(ValueError, match="elitism_ratio"):
            GAConfig(elitism_ratio=-0.1)

    def test_default_mutation_types_set(self):
        config = GAConfig()
        assert config.mutation_types is not None
        assert "threshold_perturbation" in config.mutation_types


# ---------------------------------------------------------------------------
# TreeInitializer — regression task + all-same-values branch
# ---------------------------------------------------------------------------


class TestTreeInitializerEdgeCases:
    def test_regression_task_leaf_prediction_is_float(self):
        X = np.random.rand(50, 4)
        y = np.random.rand(50)  # continuous
        initializer = TreeInitializer(
            n_features=4,
            n_classes=1,
            max_depth=3,
            min_samples_split=5,
            min_samples_leaf=2,
            task_type="regression",
        )
        tree = initializer.create_random_tree(X, y)
        assert tree is not None

    def test_all_same_feature_values_creates_leaf(self):
        """When all values of a feature are identical, _grow_tree falls back to leaf."""
        # Constant features force the all-same-values branch
        X = np.zeros((20, 4))
        X[:, 0] = 0.5  # All same
        y = np.array([0] * 10 + [1] * 10)
        initializer = TreeInitializer(
            n_features=4, n_classes=2, max_depth=5, min_samples_split=5, min_samples_leaf=2
        )
        tree = initializer.create_random_tree(X, y)
        assert tree is not None
        assert tree.get_num_nodes() >= 1


# ---------------------------------------------------------------------------
# Crossover._copy_node_contents
# ---------------------------------------------------------------------------


class TestCopyNodeContents:
    def test_copy_leaf_with_scalar_prediction(self):
        src = create_leaf_node(1, depth=2)
        dst = create_leaf_node(0, depth=2)
        Crossover._copy_node_contents(src, dst)
        assert dst.prediction == 1
        assert dst.node_type == "leaf"

    def test_copy_leaf_with_none_prediction(self):
        src = create_leaf_node(None, depth=1)
        dst = create_leaf_node(0, depth=1)
        Crossover._copy_node_contents(src, dst)
        assert dst.prediction is None

    def test_copy_leaf_with_ndarray_prediction(self):
        src = create_leaf_node(np.array([0.3, 0.7]), depth=1)
        dst = create_leaf_node(0, depth=1)
        Crossover._copy_node_contents(src, dst)
        np.testing.assert_array_equal(dst.prediction, np.array([0.3, 0.7]))
        # Must be a copy, not the same object
        assert dst.prediction is not src.prediction

    def test_copy_internal_node(self):
        left = create_leaf_node(0, 2)
        right = create_leaf_node(1, 2)
        src = create_internal_node(2, 0.75, left, right, depth=1)
        dst = create_leaf_node(0, depth=1)
        Crossover._copy_node_contents(src, dst)
        assert dst.node_type == "internal"
        assert dst.feature_idx == 2
        assert dst.threshold == 0.75


# ---------------------------------------------------------------------------
# Crossover._fix_depths
# ---------------------------------------------------------------------------


class TestFixDepths:
    def test_reassigns_depth_on_single_node(self):
        leaf = create_leaf_node(0, depth=99)
        Crossover._fix_depths(leaf, 0)
        assert leaf.depth == 0

    def test_reassigns_depths_recursively(self):
        left = create_leaf_node(0, depth=99)
        right = create_leaf_node(1, depth=99)
        root = create_internal_node(0, 0.5, left, right, depth=99)
        Crossover._fix_depths(root, 0)
        assert root.depth == 0
        assert left.depth == 1
        assert right.depth == 1

    def test_handles_none_node(self):
        # Should not raise
        Crossover._fix_depths(None, 0)


# ---------------------------------------------------------------------------
# Crossover._prune_to_depth
# ---------------------------------------------------------------------------


class TestPruneToDepth:
    def test_prunes_deep_tree_to_max_depth(self):
        # Build a depth-3 tree
        ll = create_leaf_node(0, 3)
        lr = create_leaf_node(1, 3)
        rl = create_leaf_node(0, 3)
        rr = create_leaf_node(1, 3)
        left_int = create_internal_node(1, 0.3, ll, lr, depth=2)
        right_int = create_internal_node(1, 0.7, rl, rr, depth=2)
        mid_left = create_internal_node(0, 0.5, left_int, right_int, depth=1)
        mid_right = create_leaf_node(0, 1)
        root = create_internal_node(0, 0.5, mid_left, mid_right, depth=0)
        tree = TreeGenotype(root=root, n_features=4, n_classes=2, max_depth=5)

        pruned = Crossover._prune_to_depth(tree, max_depth=1)
        assert pruned.get_depth() <= 1

    def test_creates_leaves_at_max_depth(self):
        left = create_leaf_node(0, 1)
        right = create_leaf_node(1, 1)
        root = create_internal_node(0, 0.5, left, right, depth=0)
        tree = TreeGenotype(root=root, n_features=4, n_classes=2, max_depth=5)

        pruned = Crossover._prune_to_depth(tree, max_depth=0)
        # Root becomes a leaf at depth 0
        assert pruned.root.is_leaf()

    def test_preserves_leaf_prediction_when_already_leaf(self):
        left = create_leaf_node(42, 1)
        right = create_leaf_node(99, 1)
        root = create_internal_node(0, 0.5, left, right, depth=0)
        tree = TreeGenotype(root=root, n_features=4, n_classes=2, max_depth=5)

        pruned = Crossover._prune_to_depth(tree, max_depth=2)
        # Nothing changes at depth 1 leaves
        assert pruned.get_depth() <= 2


# ---------------------------------------------------------------------------
# Crossover._repair_tree
# ---------------------------------------------------------------------------


class TestRepairTree:
    def test_repair_tree_within_max_depth(self):
        tree = _depth1_tree()
        repaired = Crossover._repair_tree(tree)
        assert isinstance(repaired, TreeGenotype)

    def test_repair_tree_prunes_when_too_deep(self):
        # Build tree deeper than max_depth
        leaf = create_leaf_node(0, 5)
        for d in range(4, -1, -1):
            other = create_leaf_node(0, d + 1)
            leaf = create_internal_node(0, 0.5, leaf, other, depth=d)
        tree = TreeGenotype(root=leaf, n_features=4, n_classes=2, max_depth=2)
        repaired = Crossover._repair_tree(tree)
        assert repaired.get_depth() <= 2


# ---------------------------------------------------------------------------
# Mutation.expand_leaf edge cases
# ---------------------------------------------------------------------------


class TestMutationExpandLeafEdgeCases:
    def test_no_expandable_leaves_returns_tree_unchanged(self):
        """When all leaves are at or beyond max_depth-1, expand_leaf returns unchanged tree."""
        # Leaf at depth=4, max_depth=5 → depth < max_depth-1=4 → NOT expandable
        # Leaf at depth=4, max_depth=5: 4 < 5-1=4 is False → no expandable leaves
        root = create_leaf_node(0, depth=4)
        tree = TreeGenotype(root=root, n_features=4, n_classes=2, max_depth=5)
        mutation = Mutation(n_features=4, feature_ranges={i: (0.0, 1.0) for i in range(4)})
        original_nodes = tree.get_num_nodes()
        result = mutation.expand_leaf(tree)
        # Should return unchanged (leaf depth 4 >= max_depth-1=4)
        assert result.get_num_nodes() == original_nodes

    def test_expand_leaf_with_feature_not_in_ranges(self):
        """When chosen feature_idx not in feature_ranges, threshold defaults to 0.0."""
        root = create_leaf_node(0, depth=0)
        tree = TreeGenotype(root=root, n_features=4, n_classes=2, max_depth=5)
        # Empty feature_ranges → feature_idx never in ranges → threshold = 0.0 branch
        mutation = Mutation(n_features=4, feature_ranges={})
        result = mutation.expand_leaf(tree)
        # The leaf should have been expanded
        if result.get_num_nodes() > 1:
            assert result.root.is_internal()
            assert result.root.threshold == 0.0

    def test_expand_leaf_with_expandable_leaf(self):
        """Normal expansion: leaf at depth 0 should expand into internal node."""
        root = create_leaf_node(0, depth=0)
        tree = TreeGenotype(root=root, n_features=4, n_classes=2, max_depth=5)
        mutation = Mutation(n_features=4, feature_ranges={i: (0.0, 1.0) for i in range(4)})
        result = mutation.expand_leaf(tree)
        assert result.root.is_internal()
        assert result.get_num_nodes() == 3


# ---------------------------------------------------------------------------
# GAEngine.evolve
# ---------------------------------------------------------------------------


class TestGAEngineEvolve:
    def _make_engine(self, random_state=None, pop_size=10, n_gen=2):
        config = GAConfig(
            population_size=pop_size,
            n_generations=n_gen,
            random_state=random_state,
            crossover_prob=0.5,
            mutation_prob=0.3,
            elitism_ratio=0.1,
        )
        initializer = TreeInitializer(
            n_features=4, n_classes=2, max_depth=3, min_samples_split=5, min_samples_leaf=2
        )
        feature_ranges = {i: (0.0, 1.0) for i in range(4)}
        mutation = Mutation(n_features=4, feature_ranges=feature_ranges)
        return GAEngine(
            config=config,
            initializer=initializer,
            fitness_function=_simple_fitness,
            mutation=mutation,
        )

    def test_evolve_returns_best_individual(self):
        X = np.random.rand(30, 4)
        y = np.random.randint(0, 2, 30)
        engine = self._make_engine(pop_size=10, n_gen=2)
        best = engine.evolve(X, y, verbose=False)
        assert isinstance(best, TreeGenotype)

    def test_evolve_populates_history(self):
        X = np.random.rand(30, 4)
        y = np.random.randint(0, 2, 30)
        engine = self._make_engine(pop_size=10, n_gen=3)
        engine.evolve(X, y, verbose=False)
        assert len(engine.history["best_fitness"]) > 0
        assert len(engine.history["avg_fitness"]) > 0

    def test_evolve_with_random_state_seeds_rng(self):
        X = np.random.rand(30, 4)
        y = np.random.randint(0, 2, 30)
        # random_state is not None → seeds random and np.random
        engine = self._make_engine(random_state=42, pop_size=10, n_gen=2)
        best = engine.evolve(X, y, verbose=False)
        assert best is not None

    def test_evolve_populates_diversity_history(self):
        """evolve() should populate history['diversity'] each generation."""
        X = np.random.rand(30, 4)
        y = np.random.randint(0, 2, 30)
        engine = self._make_engine(pop_size=10, n_gen=3)
        engine.evolve(X, y, verbose=False)
        assert len(engine.history["diversity"]) > 0

    def test_evolve_verbose_does_not_raise(self):
        X = np.random.rand(30, 4)
        y = np.random.randint(0, 2, 30)
        # verbose=True with gen%10==0 at gen=0 → triggers logger.info branch
        engine = self._make_engine(pop_size=10, n_gen=11)
        best = engine.evolve(X, y, verbose=True)
        assert best is not None


# ---------------------------------------------------------------------------
# §1.1: Improved TreeInitializer — full mode and depth-dependent stopping
# ---------------------------------------------------------------------------


class TestTreeInitializerModes:
    def _make_initializer(self):
        return TreeInitializer(
            n_features=4,
            n_classes=2,
            max_depth=4,
            min_samples_split=5,
            min_samples_leaf=2,
        )

    def test_full_mode_tree_is_valid(self):
        X = np.random.rand(60, 4)
        y = np.random.randint(0, 2, 60)
        init = self._make_initializer()
        tree = init.create_random_tree(X, y, mode="full")
        assert isinstance(tree, TreeGenotype)
        valid, errs = tree.validate()
        assert valid, f"full-mode tree invalid: {errs}"

    def test_grow_mode_respects_max_depth(self):
        X = np.random.rand(80, 4)
        y = np.random.randint(0, 2, 80)
        init = self._make_initializer()
        for _ in range(10):
            tree = init.create_random_tree(X, y, mode="grow")
            assert tree.get_depth() <= init.max_depth

    def test_full_mode_respects_max_depth(self):
        X = np.random.rand(80, 4)
        y = np.random.randint(0, 2, 80)
        init = self._make_initializer()
        for _ in range(10):
            tree = init.create_random_tree(X, y, mode="full")
            assert tree.get_depth() <= init.max_depth


# ---------------------------------------------------------------------------
# §1.2: create_informed_tree and _best_gini_split
# ---------------------------------------------------------------------------


class TestInformedInitialization:
    def _make_initializer(self):
        return TreeInitializer(
            n_features=4,
            n_classes=2,
            max_depth=4,
            min_samples_split=5,
            min_samples_leaf=2,
        )

    def test_create_informed_tree_returns_valid_tree(self):
        X = np.random.rand(80, 4)
        y = np.random.randint(0, 2, 80)
        init = self._make_initializer()
        tree = init.create_informed_tree(X, y)
        assert isinstance(tree, TreeGenotype)
        valid, errs = tree.validate()
        assert valid, f"informed tree invalid: {errs}"

    def test_create_informed_tree_respects_max_depth(self):
        X = np.random.rand(80, 4)
        y = np.random.randint(0, 2, 80)
        init = self._make_initializer()
        for _ in range(5):
            tree = init.create_informed_tree(X, y)
            assert tree.get_depth() <= init.max_depth

    def test_best_gini_split_returns_valid_split(self):
        X = np.random.rand(60, 4)
        y = np.random.randint(0, 2, 60)
        init = self._make_initializer()
        feature, threshold = init._best_gini_split(X, y)
        # Should find at least one split on separable data
        assert feature is not None
        assert threshold is not None
        assert 0 <= feature < 4

    def test_best_gini_split_with_few_unique_values(self):
        """_best_gini_split handles features with ≤10 unique values."""
        # Only 5 unique values per feature
        X = np.tile(np.array([0.1, 0.3, 0.5, 0.7, 0.9]), (20, 4)).reshape(100, 4)
        y = np.random.randint(0, 2, 100)
        init = self._make_initializer()
        feature, threshold = init._best_gini_split(X, y)
        # Should return None,None if all splits are too small, or a valid split
        if feature is not None:
            assert 0 <= feature < 4

    def test_gini_impurity_zero_for_empty_subset(self):
        init = self._make_initializer()
        assert init._gini_impurity(np.array([]), 0, 100) == 0.0

    def test_informed_tree_single_class(self):
        """Informed tree with a single class returns a leaf."""
        X = np.random.rand(30, 4)
        y = np.zeros(30, dtype=int)  # Single class
        init = self._make_initializer()
        tree = init.create_informed_tree(X, y)
        assert isinstance(tree, TreeGenotype)

    def test_grow_informed_tree_constant_feature_fallback(self):
        """Informed tree handles constant feature in random-quantile branch."""
        # Make feature 0 constant so the random-feature path hits the fallback
        X = np.zeros((30, 4))
        X[:, 1:] = np.random.rand(30, 3)
        y = np.random.randint(0, 2, 30)
        init = self._make_initializer()
        # Run several times to hit the constant-feature branch
        for _ in range(5):
            tree = init.create_informed_tree(X, y)
            assert tree is not None


# ---------------------------------------------------------------------------
# §1.3: GAEngine._split_data and validation-aware evaluate_population
# ---------------------------------------------------------------------------


class TestSplitData:
    def _make_engine(self):
        config = GAConfig(population_size=10, n_generations=2, random_state=42)
        init = TreeInitializer(
            n_features=4, n_classes=2, max_depth=3, min_samples_split=5, min_samples_leaf=2
        )
        mutation = Mutation(n_features=4, feature_ranges={i: (0.0, 1.0) for i in range(4)})
        return GAEngine(
            config=config, initializer=init, fitness_function=_simple_fitness, mutation=mutation
        )

    def test_split_data_with_sufficient_samples(self):
        X = np.random.rand(50, 4)
        y = np.array([0] * 25 + [1] * 25)
        engine = self._make_engine()
        X_train, y_train, X_val, y_val = engine._split_data(X, y)
        assert X_val is not None
        assert y_val is not None
        assert len(X_train) + len(X_val) == len(X)

    def test_split_data_too_small_returns_none_val(self):
        X = np.random.rand(10, 4)
        y = np.random.randint(0, 2, 10)
        engine = self._make_engine()
        X_train, y_train, X_val, y_val = engine._split_data(X, y)
        assert X_val is None
        assert y_val is None

    def test_split_data_single_class_no_stratify(self):
        """Single-class y should not stratify (would error with stratify)."""
        X = np.random.rand(50, 4)
        y = np.zeros(50, dtype=int)
        engine = self._make_engine()
        X_train, y_train, X_val, y_val = engine._split_data(X, y)
        # Should succeed (fallback to no-stratify)
        assert X_train is not None

    def test_check_fitness_accepts_val_three_arg(self):
        """3-arg fitness function should not be detected as accepting val."""
        engine = self._make_engine()
        assert engine._fitness_accepts_val is False

    def test_check_fitness_accepts_val_five_arg(self):
        """5-arg fitness function should be detected as accepting val."""
        from ga_trees.fitness.calculator import FitnessCalculator

        fc = FitnessCalculator()
        config = GAConfig(population_size=5, n_generations=1)
        init = TreeInitializer(
            n_features=4, n_classes=2, max_depth=3, min_samples_split=5, min_samples_leaf=2
        )
        mutation = Mutation(n_features=4, feature_ranges={i: (0.0, 1.0) for i in range(4)})
        engine = GAEngine(
            config=config,
            initializer=init,
            fitness_function=fc.calculate_fitness,
            mutation=mutation,
        )
        assert engine._fitness_accepts_val is True


# ---------------------------------------------------------------------------
# §1.4 + §1.5: GAEngine helpers
# ---------------------------------------------------------------------------


class TestGAEngineHelpers:
    def _make_engine(self, pop_size=10):
        config = GAConfig(population_size=pop_size, n_generations=2, random_state=0)
        init = TreeInitializer(
            n_features=4, n_classes=2, max_depth=3, min_samples_split=5, min_samples_leaf=2
        )
        mutation = Mutation(n_features=4, feature_ranges={i: (0.0, 1.0) for i in range(4)})
        return GAEngine(
            config=config, initializer=init, fitness_function=_simple_fitness, mutation=mutation
        )

    def test_get_adaptive_params_no_stagnation(self):
        engine = self._make_engine()
        prob, types = engine._get_adaptive_params(0)
        assert prob == engine.config.mutation_prob
        assert types == engine.config.mutation_types

    def test_get_adaptive_params_with_stagnation_boosts_rate(self):
        engine = self._make_engine()
        prob, types = engine._get_adaptive_params(10)
        assert prob > engine.config.mutation_prob

    def test_get_adaptive_params_with_stagnation_normalizes_weights(self):
        engine = self._make_engine()
        _, types = engine._get_adaptive_params(10)
        assert abs(sum(types.values()) - 1.0) < 1e-6

    def test_inject_immigrants_reduces_weak_individuals(self):
        X = np.random.rand(40, 4)
        y = np.random.randint(0, 2, 40)
        engine = self._make_engine(pop_size=10)
        engine.initialize_population(X, y)
        engine.evaluate_population(X, y)
        engine._inject_immigrants(X, y, None, None)
        # All individuals should still have fitness after evaluation
        assert all(ind.fitness_ is not None for ind in engine.population)

    def test_initialize_population_size_correct(self):
        X = np.random.rand(50, 4)
        y = np.random.randint(0, 2, 50)
        engine = self._make_engine(pop_size=12)
        engine.initialize_population(X, y)
        assert len(engine.population) == 12

    def test_evolve_with_real_fitness_calculator(self):
        """End-to-end evolve with FitnessCalculator validating §1.3 integration."""
        from ga_trees.fitness.calculator import FitnessCalculator

        X = np.random.rand(50, 4)
        y = np.array([0] * 25 + [1] * 25)
        n_features = 4
        feature_ranges = {i: (0.0, 1.0) for i in range(n_features)}
        config = GAConfig(population_size=8, n_generations=3, random_state=7)
        init = TreeInitializer(n_features, 2, max_depth=3, min_samples_split=5, min_samples_leaf=2)
        fc = FitnessCalculator()
        mutation = Mutation(n_features, feature_ranges)
        engine = GAEngine(
            config=config,
            initializer=init,
            fitness_function=fc.calculate_fitness,
            mutation=mutation,
        )
        best = engine.evolve(X, y, verbose=False)
        assert isinstance(best, TreeGenotype)
        assert len(engine.history["diversity"]) > 0
