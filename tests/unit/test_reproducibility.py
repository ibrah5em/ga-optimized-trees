"""Tests for deterministic seeding (LDD-9).

The claim "random seed: 42 (reproducibility)" was published for a year while
``scripts/experiment.py`` built ``GAConfig`` without ``random_state``, so no GA
run was actually reproducible. These tests make the claim checkable.
"""

import numpy as np
import pytest
from sklearn.datasets import load_iris

from ga_trees.fitness.calculator import FitnessCalculator
from ga_trees.ga.engine import GAConfig, GAEngine, Mutation, TreeInitializer
from ga_trees.reproducibility import MAX_SEED, build_seed_manifest, derive_fold_seed


class TestDeriveFoldSeed:
    """Seed derivation must be stable, distinct, and in range."""

    def test_is_deterministic_across_calls(self):
        assert derive_fold_seed(42, "iris", 1) == derive_fold_seed(42, "iris", 1)

    def test_differs_across_folds(self):
        seeds = [derive_fold_seed(42, "iris", fold) for fold in range(1, 11)]
        assert len(set(seeds)) == len(seeds)

    def test_differs_across_datasets(self):
        assert derive_fold_seed(42, "iris", 1) != derive_fold_seed(42, "wine", 1)

    def test_differs_across_methods(self):
        ga = derive_fold_seed(42, "iris", 1, method="ga")
        rs = derive_fold_seed(42, "iris", 1, method="random_search")
        assert ga != rs

    def test_differs_across_base_seeds(self):
        assert derive_fold_seed(42, "iris", 1) != derive_fold_seed(43, "iris", 1)

    def test_within_valid_range(self):
        for fold in range(1, 51):
            seed = derive_fold_seed(7, "breast_cancer", fold)
            assert 0 <= seed < MAX_SEED

    def test_accepted_by_numpy(self):
        # The whole point is that these values can seed the RNGs the GA uses.
        np.random.seed(derive_fold_seed(42, "iris", 3))

    def test_does_not_depend_on_python_hash_salt(self):
        """Guards against a hash()-based implementation.

        Python salts str hashing per process, so a hash() derivation would give
        different seeds on each invocation. These literals were captured from
        the blake2b implementation; if they drift, the derivation changed and
        every previously recorded seeds.json is no longer reproducible.
        """
        assert derive_fold_seed(42, "iris", 1) == 2053655277
        assert derive_fold_seed(42, "wine", 5, method="random_search") == 353924766

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"base_seed": -1, "dataset_name": "iris", "fold": 1},
            {"base_seed": 42, "dataset_name": "iris", "fold": -1},
            {"base_seed": 42, "dataset_name": "", "fold": 1},
        ],
    )
    def test_rejects_invalid_input(self, kwargs):
        with pytest.raises(ValueError):
            derive_fold_seed(**kwargs)


class TestSeedManifest:
    def test_lists_one_seed_per_fold(self):
        manifest = build_seed_manifest(42, ["iris", "wine"], n_folds=5)
        assert manifest["base_seed"] == 42
        assert sorted(manifest["folds"]) == ["iris", "wine"]
        assert len(manifest["folds"]["iris"]["ga"]) == 5

    def test_matches_derive_fold_seed(self):
        manifest = build_seed_manifest(42, ["iris"], n_folds=3)
        expected = [derive_fold_seed(42, "iris", fold) for fold in (1, 2, 3)]
        assert manifest["folds"]["iris"]["ga"] == expected

    def test_supports_multiple_methods(self):
        manifest = build_seed_manifest(42, ["iris"], 2, methods=("ga", "random_search"))
        assert manifest["folds"]["iris"]["ga"] != manifest["folds"]["iris"]["random_search"]

    def test_rejects_zero_folds(self):
        with pytest.raises(ValueError):
            build_seed_manifest(42, ["iris"], n_folds=0)


def _evolve(seed, X, y):
    """Run a short evolution at a given seed and return the champion tree."""
    n_features = X.shape[1]
    config = GAConfig(
        population_size=12,
        n_generations=3,
        mutation_types={
            "threshold_perturbation": 0.4,
            "feature_replacement": 0.3,
            "prune_subtree": 0.2,
            "expand_leaf": 0.1,
        },
        random_state=seed,
    )
    initializer = TreeInitializer(
        n_features=n_features,
        n_classes=len(np.unique(y)),
        max_depth=4,
        min_samples_split=10,
        min_samples_leaf=5,
    )
    fitness = FitnessCalculator()
    mutation = Mutation(
        n_features=n_features,
        feature_ranges={i: (X[:, i].min(), X[:, i].max()) for i in range(n_features)},
    )
    engine = GAEngine(config, initializer, fitness.calculate_fitness, mutation)
    return engine.evolve(X, y, verbose=False)


class TestSeededEvolutionIsReproducible:
    """A seeded GA run must be repeatable - the claim that was false for a year."""

    @pytest.fixture(scope="class")
    def data(self):
        X, y = load_iris(return_X_y=True)
        return X, y

    def test_same_seed_gives_identical_tree(self, data):
        X, y = data
        first = _evolve(4242, X, y)
        second = _evolve(4242, X, y)
        assert first.to_dict() == second.to_dict()

    def test_same_seed_gives_identical_fitness(self, data):
        X, y = data
        assert _evolve(4242, X, y).fitness_ == _evolve(4242, X, y).fitness_

    def test_unseeded_runs_are_not_forced_equal(self, data):
        """random_state=None must not silently behave as a fixed seed."""
        X, y = data
        trees = [_evolve(None, X, y).to_dict() for _ in range(4)]
        assert any(t != trees[0] for t in trees[1:])

    def test_derived_fold_seeds_give_different_trees(self, data):
        """Per-fold seeds must actually decorrelate folds.

        If every fold reused one base seed, each fold would repeat the same
        search and the cross-validation would report one run five times.
        """
        X, y = data
        trees = [
            _evolve(derive_fold_seed(42, "iris", fold), X, y).to_dict() for fold in range(1, 4)
        ]
        assert any(t != trees[0] for t in trees[1:])
