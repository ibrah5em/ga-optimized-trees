"""Unit tests for validation-based fitness.

Fitness used to be resubstitution: leaf predictions were fitted on the same rows
the tree was then scored on, so the search ranked individuals by how well they
memorised the fitting set. These tests pin down that the GA-validation split is
held out from every part of the search — initialization, thresholds and scoring —
and that random search is treated identically, so the comparison stays fair.
"""

import numpy as np
import pytest
from sklearn.datasets import load_iris

from ga_trees.benchmark.methods import (
    GATreeMethod,
    RandomTreeSearch,
    _build_search_context,
    holdout_split,
)
from ga_trees.fitness.calculator import FitnessCalculator
from ga_trees.ga.engine import GAConfig, GAEngine, Mutation, TreeInitializer

TREE_CONFIG = {
    "max_depth": 4,
    "min_samples_split": 5,
    "min_samples_leaf": 2,
    "growth_stop_prob": 0.3,
}
GA_CONFIG = {
    "population_size": 8,
    "n_generations": 2,
    "crossover_prob": 0.7,
    "mutation_prob": 0.2,
    "tournament_size": 3,
    "elitism_ratio": 0.25,
    "mutation_types": None,
}
FITNESS_CONFIG = {
    "weights": {"accuracy": 0.7, "interpretability": 0.3},
    "validation_fraction": 0.25,
}


@pytest.fixture
def iris():
    return load_iris(return_X_y=True)


class TestHoldoutSplit:
    def test_split_sizes_and_disjointness(self, iris):
        X, y = iris
        X_tr, y_tr, X_val, y_val = holdout_split(X, y, fraction=0.2, seed=0)
        assert len(X_tr) + len(X_val) == len(X)
        assert len(X_val) == pytest.approx(len(X) * 0.2, abs=1)
        assert len(y_tr) == len(X_tr) and len(y_val) == len(X_val)

    def test_split_is_stratified(self, iris):
        X, y = iris
        _, y_tr, _, y_val = holdout_split(X, y, fraction=0.2, seed=0)
        train_share = np.bincount(y_tr) / len(y_tr)
        val_share = np.bincount(y_val) / len(y_val)
        assert train_share == pytest.approx(val_share, abs=0.05)

    def test_zero_fraction_disables_the_split(self, iris):
        X, y = iris
        X_tr, y_tr, X_val, y_val = holdout_split(X, y, fraction=0.0, seed=0)
        assert X_val is None and y_val is None
        assert X_tr is X and y_tr is y

    def test_same_seed_gives_the_same_split(self, iris):
        X, y = iris
        first = holdout_split(X, y, fraction=0.2, seed=7)
        second = holdout_split(X, y, fraction=0.2, seed=7)
        assert np.array_equal(first[2], second[2])

    def test_falls_back_when_a_class_is_too_small_to_stratify(self):
        X = np.arange(20, dtype=float).reshape(10, 2)
        y = np.array([0] * 9 + [1])  # one member in class 1
        X_tr, _, X_val, y_val = holdout_split(X, y, fraction=0.2, seed=0)
        assert X_val is None and y_val is None
        assert X_tr is X

    def test_rejects_a_fraction_outside_the_unit_interval(self, iris):
        X, y = iris
        with pytest.raises(ValueError, match="validation_fraction"):
            holdout_split(X, y, fraction=1.5, seed=0)


class TestEngineValidationWiring:
    def _engine(self, X, fitness_function):
        return GAEngine(
            config=GAConfig(population_size=6, n_generations=1, random_state=0),
            initializer=TreeInitializer(
                n_features=X.shape[1],
                n_classes=3,
                max_depth=3,
                min_samples_split=5,
                min_samples_leaf=2,
            ),
            fitness_function=fitness_function,
            mutation=Mutation(
                n_features=X.shape[1],
                feature_ranges={j: (0.0, 10.0) for j in range(X.shape[1])},
            ),
        )

    def test_validation_arrays_reach_the_fitness_function(self, iris):
        X, y = iris
        X_tr, y_tr, X_val, y_val = holdout_split(X, y, fraction=0.2, seed=0)
        seen = []

        def spy(tree, X_fit, y_fit, X_score=None, y_score=None):
            seen.append((X_fit.shape, None if X_score is None else X_score.shape))
            return 0.5

        self._engine(X, spy).evolve(X_tr, y_tr, verbose=False, X_val=X_val, y_val=y_val)
        assert seen
        assert all(fit == X_tr.shape and score == X_val.shape for fit, score in seen)

    def test_three_argument_fitness_functions_still_work(self, iris):
        X, y = iris
        calls = []

        def legacy(tree, X_fit, y_fit):
            calls.append(X_fit.shape)
            return 0.5

        self._engine(X, legacy).evolve(X, y, verbose=False)
        assert calls and all(shape == X.shape for shape in calls)

    def test_half_a_validation_set_is_rejected(self, iris):
        X, y = iris
        engine = self._engine(X, lambda t, a, b: 0.0)
        with pytest.raises(ValueError, match="together"):
            engine.evolve(X, y, verbose=False, X_val=X)

    def test_population_is_seeded_from_training_data_only(self, iris):
        X, y = iris
        X_tr, y_tr, X_val, y_val = holdout_split(X, y, fraction=0.2, seed=0)
        seen_shapes = []

        class SpyInitializer(TreeInitializer):
            def create_random_tree(self, X_seed, y_seed):
                seen_shapes.append(X_seed.shape)
                return super().create_random_tree(X_seed, y_seed)

        engine = GAEngine(
            config=GAConfig(population_size=4, n_generations=1, random_state=0),
            initializer=SpyInitializer(X.shape[1], 3, 3, 5, 2),
            fitness_function=lambda t, a, b, c=None, d=None: 0.5,
            mutation=Mutation(X.shape[1], {j: (0.0, 10.0) for j in range(X.shape[1])}),
        )
        engine.evolve(X_tr, y_tr, verbose=False, X_val=X_val, y_val=y_val)
        assert seen_shapes and all(shape == X_tr.shape for shape in seen_shapes)

    def test_champion_is_the_best_on_validation_not_on_training(self, iris):
        X, y = iris
        X_tr, y_tr, X_val, y_val = holdout_split(X, y, fraction=0.25, seed=0)
        calculator = FitnessCalculator(mode="weighted_sum")

        engine = self._engine(X, calculator.calculate_fitness)
        best = engine.evolve(X_tr, y_tr, verbose=False, X_val=X_val, y_val=y_val)

        # The stored fitness must be reproducible from the validation split.
        expected = calculator.calculate_fitness(best, X_tr, y_tr, X_val, y_val)
        assert best.fitness_ == pytest.approx(expected)

    def test_validation_fitness_is_not_the_resubstitution_fitness(self, iris):
        # The whole point: scoring on held-out rows must be capable of
        # disagreeing with scoring on the fitted rows.
        X, y = iris
        X_tr, y_tr, X_val, y_val = holdout_split(X, y, fraction=0.3, seed=1)
        calculator = FitnessCalculator(mode="weighted_sum")
        initializer = TreeInitializer(X.shape[1], 3, 5, 5, 2, growth_stop_prob=0.05)

        np.random.seed(0)
        import random

        random.seed(0)
        differing = 0
        for _ in range(20):
            tree = initializer.create_random_tree(X_tr, y_tr)
            resubstitution = calculator.calculate_fitness(tree, X_tr, y_tr)
            validation = calculator.calculate_fitness(tree, X_tr, y_tr, X_val, y_val)
            if resubstitution != validation:
                differing += 1
        assert differing > 0


class TestBudgetMatchedMethodsShareTheSplit:
    def test_ga_and_random_search_get_identical_contexts(self, iris):
        X, y = iris
        params = {"accuracy_weight": 0.7, "max_depth": 4}
        ga = _build_search_context(TREE_CONFIG, FITNESS_CONFIG, X, y, params, seed=11)
        rs = _build_search_context(TREE_CONFIG, FITNESS_CONFIG, X, y, params, seed=11)

        assert np.array_equal(ga.X_train, rs.X_train)
        assert np.array_equal(ga.X_val, rs.X_val)
        assert np.array_equal(ga.y_val, rs.y_val)
        assert ga.initializer.max_depth == rs.initializer.max_depth
        assert ga.initializer.split_strategy == rs.initializer.split_strategy

    def test_validation_split_does_not_change_the_evaluation_budget(self, iris):
        # The GA vs random search comparison needs both to spend the same number of
        # evaluations, with or without the holdout.
        X, y = iris
        without = dict(FITNESS_CONFIG, validation_fraction=0.0)

        counts = {}
        for label, fitness_config in (("val", FITNESS_CONFIG), ("resub", without)):
            for method in (
                GATreeMethod(GA_CONFIG, TREE_CONFIG, fitness_config, tune=False),
                RandomTreeSearch(GA_CONFIG, TREE_CONFIG, fitness_config, tune=False),
            ):
                model = method.fit(X, y, {}, seed=3)
                counts[(label, method.name)] = model.n_evaluations

        assert counts[("val", "GA-Optimized")] == counts[("resub", "GA-Optimized")]
        assert counts[("val", "Random Search")] == counts[("resub", "Random Search")]
        assert counts[("val", "GA-Optimized")] == counts[("val", "Random Search")]

    @pytest.mark.parametrize("fraction", [0.0, 0.25])
    def test_both_methods_still_produce_usable_models(self, iris, fraction):
        X, y = iris
        fitness_config = dict(FITNESS_CONFIG, validation_fraction=fraction)
        for method in (
            GATreeMethod(GA_CONFIG, TREE_CONFIG, fitness_config, tune=False),
            RandomTreeSearch(GA_CONFIG, TREE_CONFIG, fitness_config, tune=False),
        ):
            model = method.fit(X, y, {}, seed=5)
            predictions = model.predict(X)
            assert predictions.shape == y.shape
            assert model.n_leaves >= 1
