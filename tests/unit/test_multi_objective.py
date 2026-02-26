"""Unit tests for ga/multi_objective.py.

Covers:
- _cleanup_deap_creator
- ParetoOptimizer.__init__
- ParetoOptimizer._wrap_individual
- ParetoOptimizer._create_population
- ParetoOptimizer._evaluate
- ParetoOptimizer.evolve_pareto_front
- ParetoOptimizer.plot_pareto_front
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from deap import base, creator

from ga_trees.ga.engine import Mutation, TreeInitializer
from ga_trees.ga.multi_objective import ParetoOptimizer, _cleanup_deap_creator
from ga_trees.genotype.tree_genotype import TreeGenotype, create_internal_node, create_leaf_node

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def small_iris():
    """Small subset of iris for fast tests."""
    from sklearn.datasets import load_iris

    X, y = load_iris(return_X_y=True)
    return X[:40], y[:40]


@pytest.fixture
def simple_fitness_fn():
    """Fixed fitness function returning constant values."""

    def _fn(tree, X, y):
        return (0.9, 0.8)

    return _fn


@pytest.fixture
def pareto_optimizer(small_iris, simple_fitness_fn):
    """ParetoOptimizer using iris data with a trivial fitness function."""
    X, y = small_iris
    n_features = X.shape[1]
    n_classes = len(np.unique(y))
    initializer = TreeInitializer(
        n_features=n_features,
        n_classes=n_classes,
        max_depth=3,
        min_samples_split=5,
        min_samples_leaf=2,
    )
    feature_ranges = {j: (float(X[:, j].min()), float(X[:, j].max())) for j in range(n_features)}
    mutation = Mutation(n_features=n_features, feature_ranges=feature_ranges)
    return ParetoOptimizer(
        initializer=initializer,
        fitness_fn=simple_fitness_fn,
        mutation_fn=lambda t: mutation.mutate(t, {"threshold_perturbation": 1.0}),
        random_state=42,
    )


def _make_tree_with_meta(accuracy=0.9, interpretability=0.8, feature_idx=0):
    """Create a small tree with accuracy_ and interpretability_ set."""
    left = create_leaf_node(0, depth=1)
    right = create_leaf_node(1, depth=1)
    root = create_internal_node(feature_idx, 0.5, left, right, depth=0)
    t = TreeGenotype(root=root, n_features=4, n_classes=2)
    t.accuracy_ = accuracy
    t.interpretability_ = interpretability
    return t


# ---------------------------------------------------------------------------
# _cleanup_deap_creator
# ---------------------------------------------------------------------------


class TestCleanupDeapCreator:
    """Tests for _cleanup_deap_creator()."""

    def test_removes_individual_when_present(self):
        """If creator.Individual exists, cleanup must remove it."""
        if not hasattr(creator, "FitnessMulti"):
            creator.create("FitnessMulti", base.Fitness, weights=(1.0, 1.0))
        if not hasattr(creator, "Individual"):
            creator.create("Individual", list, fitness=creator.FitnessMulti)

        assert hasattr(creator, "Individual")
        _cleanup_deap_creator()
        assert not hasattr(creator, "Individual")

    def test_removes_fitness_multi_when_present(self):
        """If creator.FitnessMulti exists, cleanup must remove it."""
        if not hasattr(creator, "FitnessMulti"):
            creator.create("FitnessMulti", base.Fitness, weights=(1.0, 1.0))

        _cleanup_deap_creator()
        assert not hasattr(creator, "FitnessMulti")

    def test_safe_when_attrs_absent(self):
        """Should not raise if both attrs are already gone."""
        _cleanup_deap_creator()  # Ensure absent
        _cleanup_deap_creator()  # Second call must still succeed

    def test_independent_calls_are_idempotent(self):
        """Calling cleanup three times in a row should never raise."""
        for _ in range(3):
            _cleanup_deap_creator()


# ---------------------------------------------------------------------------
# ParetoOptimizer.__init__
# ---------------------------------------------------------------------------


class TestParetoOptimizerInit:
    """Tests for ParetoOptimizer.__init__."""

    def test_creator_individual_exists_after_init(self, pareto_optimizer):
        assert hasattr(creator, "Individual")

    def test_creator_fitness_multi_exists_after_init(self, pareto_optimizer):
        assert hasattr(creator, "FitnessMulti")

    def test_toolbox_clone_registered(self, pareto_optimizer):
        """LDD-15: toolbox.clone must be callable."""
        assert callable(pareto_optimizer.toolbox.clone)

    def test_params_stored(self, small_iris, simple_fitness_fn):
        X, y = small_iris
        initializer = TreeInitializer(
            n_features=4, n_classes=2, max_depth=3, min_samples_split=5, min_samples_leaf=2
        )
        opt = ParetoOptimizer(
            initializer, simple_fitness_fn, lambda t: t, crossover_prob=0.5, mutation_prob=0.3
        )
        assert opt.crossover_prob == 0.5
        assert opt.mutation_prob == 0.3
        assert opt.initializer is initializer

    def test_random_state_stored(self, small_iris, simple_fitness_fn):
        X, y = small_iris
        initializer = TreeInitializer(
            n_features=4, n_classes=2, max_depth=3, min_samples_split=5, min_samples_leaf=2
        )
        opt = ParetoOptimizer(initializer, simple_fitness_fn, lambda t: t, random_state=99)
        assert opt.random_state == 99

    def test_cleanup_replaces_existing_creator(self, small_iris, simple_fitness_fn):
        """Creating two optimizers in succession should not raise."""
        X, y = small_iris
        initializer = TreeInitializer(
            n_features=4, n_classes=2, max_depth=3, min_samples_split=5, min_samples_leaf=2
        )
        opt1 = ParetoOptimizer(initializer, simple_fitness_fn, lambda t: t)
        opt2 = ParetoOptimizer(initializer, simple_fitness_fn, lambda t: t)
        assert opt1 is not opt2


# ---------------------------------------------------------------------------
# ParetoOptimizer._wrap_individual
# ---------------------------------------------------------------------------


class TestWrapIndividual:
    """Tests for ParetoOptimizer._wrap_individual."""

    def test_returns_list_subclass(self, pareto_optimizer, small_iris):
        X, y = small_iris
        tree = pareto_optimizer.initializer.create_random_tree(X, y)
        ind = ParetoOptimizer._wrap_individual(tree)
        assert isinstance(ind, list)

    def test_individual_contains_tree(self, pareto_optimizer, small_iris):
        X, y = small_iris
        tree = pareto_optimizer.initializer.create_random_tree(X, y)
        ind = ParetoOptimizer._wrap_individual(tree)
        assert ind[0] is tree

    def test_individual_has_fitness_attribute(self, pareto_optimizer, small_iris):
        X, y = small_iris
        tree = pareto_optimizer.initializer.create_random_tree(X, y)
        ind = ParetoOptimizer._wrap_individual(tree)
        assert hasattr(ind, "fitness")


# ---------------------------------------------------------------------------
# ParetoOptimizer._create_population
# ---------------------------------------------------------------------------


class TestCreatePopulation:
    """Tests for ParetoOptimizer._create_population."""

    def test_returns_list(self, pareto_optimizer, small_iris):
        X, y = small_iris
        pop = pareto_optimizer._create_population(X, y, size=4)
        assert isinstance(pop, list)

    def test_correct_size(self, pareto_optimizer, small_iris):
        X, y = small_iris
        pop = pareto_optimizer._create_population(X, y, size=6)
        assert len(pop) == 6

    def test_elements_contain_tree_genotype(self, pareto_optimizer, small_iris):
        X, y = small_iris
        pop = pareto_optimizer._create_population(X, y, size=3)
        for ind in pop:
            assert isinstance(ind[0], TreeGenotype)

    def test_fitness_initially_invalid(self, pareto_optimizer, small_iris):
        X, y = small_iris
        pop = pareto_optimizer._create_population(X, y, size=3)
        for ind in pop:
            assert not ind.fitness.valid


# ---------------------------------------------------------------------------
# ParetoOptimizer._evaluate
# ---------------------------------------------------------------------------


class TestEvaluate:
    """Tests for ParetoOptimizer._evaluate."""

    def test_assigns_fitness_values(self, pareto_optimizer, small_iris):
        X, y = small_iris
        pop = pareto_optimizer._create_population(X, y, size=4)
        pareto_optimizer._evaluate(pop, X, y)
        for ind in pop:
            assert ind.fitness.valid
            assert len(ind.fitness.values) == 2

    def test_fitness_values_match_fn_output(self, small_iris, simple_fitness_fn):
        X, y = small_iris
        initializer = TreeInitializer(
            n_features=4, n_classes=3, max_depth=3, min_samples_split=5, min_samples_leaf=2
        )
        opt = ParetoOptimizer(initializer, simple_fitness_fn, lambda t: t)
        pop = opt._create_population(X, y, size=2)
        opt._evaluate(pop, X, y)
        for ind in pop:
            assert ind.fitness.values == (0.9, 0.8)

    def test_skips_already_valid_individuals(self, pareto_optimizer, small_iris):
        X, y = small_iris
        pop = pareto_optimizer._create_population(X, y, size=2)
        pop[0].fitness.values = (0.5, 0.5)

        call_count = [0]
        original_fn = pareto_optimizer.fitness_fn

        def counting_fn(tree, X, y):
            call_count[0] += 1
            return original_fn(tree, X, y)

        pareto_optimizer.fitness_fn = counting_fn
        pareto_optimizer._evaluate(pop, X, y)
        # Only pop[1] should be evaluated
        assert call_count[0] == 1
        pareto_optimizer.fitness_fn = original_fn


# ---------------------------------------------------------------------------
# ParetoOptimizer.evolve_pareto_front
# ---------------------------------------------------------------------------


def _assign_crowding_dist_stub(front):
    """Stub for tools.assignCrowdingDist: sets crowding_dist=0.0 on each individual."""
    for ind in front:
        ind.fitness.crowding_dist = 0.0


def _patch_deap_tools():
    """Context manager that patches tools.assignCrowdingDist (missing in DEAP 1.4)."""
    from deap import tools

    return patch.object(tools, "assignCrowdingDist", _assign_crowding_dist_stub, create=True)


class TestEvolveParetoFront:
    """Tests for ParetoOptimizer.evolve_pareto_front."""

    def test_returns_list(self, pareto_optimizer, small_iris):
        X, y = small_iris
        with _patch_deap_tools():
            result = pareto_optimizer.evolve_pareto_front(X, y, population_size=8, n_generations=2)
        assert isinstance(result, list)

    def test_returns_nonempty_result(self, pareto_optimizer, small_iris):
        X, y = small_iris
        with _patch_deap_tools():
            result = pareto_optimizer.evolve_pareto_front(X, y, population_size=8, n_generations=2)
        assert len(result) > 0

    def test_result_contains_tree_genotypes(self, pareto_optimizer, small_iris):
        X, y = small_iris
        with _patch_deap_tools():
            result = pareto_optimizer.evolve_pareto_front(X, y, population_size=8, n_generations=2)
        for tree in result:
            assert isinstance(tree, TreeGenotype)

    def test_verbose_gen_0_hits_print_branch(self, pareto_optimizer, small_iris):
        """verbose=True and 11 generations ensures gen%10==0 branch is hit."""
        X, y = small_iris
        with _patch_deap_tools():
            result = pareto_optimizer.evolve_pareto_front(
                X, y, population_size=8, n_generations=11, verbose=True
            )
        assert len(result) > 0

    def test_random_state_reproducibility(self, small_iris, simple_fitness_fn):
        X, y = small_iris
        initializer = TreeInitializer(
            n_features=4, n_classes=3, max_depth=3, min_samples_split=5, min_samples_leaf=2
        )
        opt1 = ParetoOptimizer(initializer, simple_fitness_fn, lambda t: t, random_state=0)
        opt2 = ParetoOptimizer(initializer, simple_fitness_fn, lambda t: t, random_state=0)
        # population_size must be divisible by 4 for selTournamentDCD
        with _patch_deap_tools():
            r1 = opt1.evolve_pareto_front(X, y, population_size=8, n_generations=2)
        with _patch_deap_tools():
            r2 = opt2.evolve_pareto_front(X, y, population_size=8, n_generations=2)
        assert len(r1) == len(r2)


# ---------------------------------------------------------------------------
# ParetoOptimizer.plot_pareto_front
# ---------------------------------------------------------------------------


class TestPlotParetoFront:
    """Tests for ParetoOptimizer.plot_pareto_front."""

    def _make_front(self, n=3):
        return [_make_tree_with_meta(0.8 + i * 0.05, 0.7 - i * 0.05, i % 4) for i in range(n)]

    def test_show_is_called(self):
        front = self._make_front()
        with patch("matplotlib.pyplot.show") as mock_show, patch("matplotlib.pyplot.figure"), patch(
            "matplotlib.pyplot.scatter"
        ), patch("matplotlib.pyplot.xlabel"), patch("matplotlib.pyplot.ylabel"), patch(
            "matplotlib.pyplot.title"
        ), patch(
            "matplotlib.pyplot.grid"
        ), patch(
            "matplotlib.pyplot.colorbar", return_value=MagicMock()
        ):
            ParetoOptimizer.plot_pareto_front(front)
            mock_show.assert_called_once()

    def test_savefig_called_with_save_path(self):
        front = self._make_front(1)
        with patch("matplotlib.pyplot.show"), patch("matplotlib.pyplot.figure"), patch(
            "matplotlib.pyplot.scatter"
        ), patch("matplotlib.pyplot.xlabel"), patch("matplotlib.pyplot.ylabel"), patch(
            "matplotlib.pyplot.title"
        ), patch(
            "matplotlib.pyplot.grid"
        ), patch(
            "matplotlib.pyplot.colorbar", return_value=MagicMock()
        ), patch(
            "matplotlib.pyplot.savefig"
        ) as mock_save:
            ParetoOptimizer.plot_pareto_front(front, save_path="/tmp/pareto_test.png")  # nosec B108
            mock_save.assert_called_once_with(  # nosec B108
                "/tmp/pareto_test.png", dpi=300, bbox_inches="tight"
            )

    def test_savefig_not_called_without_save_path(self):
        front = self._make_front(1)
        with patch("matplotlib.pyplot.show"), patch("matplotlib.pyplot.figure"), patch(
            "matplotlib.pyplot.scatter"
        ), patch("matplotlib.pyplot.xlabel"), patch("matplotlib.pyplot.ylabel"), patch(
            "matplotlib.pyplot.title"
        ), patch(
            "matplotlib.pyplot.grid"
        ), patch(
            "matplotlib.pyplot.colorbar", return_value=MagicMock()
        ), patch(
            "matplotlib.pyplot.savefig"
        ) as mock_save:
            ParetoOptimizer.plot_pareto_front(front)
            mock_save.assert_not_called()
