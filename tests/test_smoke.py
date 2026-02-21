"""Smoke test — verify package is importable and core workflow runs."""

import numpy as np
import pytest


def test_imports():
    """All top-level public exports must be importable from ga_trees."""
    from ga_trees import (  # noqa: F401
        Crossover,
        DatasetLoader,
        FitnessCalculator,
        GAConfig,
        GAEngine,
        Mutation,
        Selection,
        TreeGenotype,
        TreeInitializer,
    )


def test_subpackage_imports():
    """Sub-package exports must be importable directly."""
    from ga_trees.data import DatasetLoader, DataValidator, load_benchmark_dataset  # noqa
    from ga_trees.evaluation import (  # noqa
        FeatureImportanceAnalyzer,
        MetricsCalculator,
        TreeExplainer,
        TreeVisualizer,
    )
    from ga_trees.fitness import (  # noqa
        FitnessCalculator,
        InterpretabilityCalculator,
        TreePredictor,
    )
    from ga_trees.ga import (  # noqa
        Crossover,
        GAConfig,
        GAEngine,
        Mutation,
        Selection,
        TreeInitializer,
        safe_subtree_crossover,
    )
    from ga_trees.genotype import (  # noqa
        Node,
        TreeGenotype,
        create_internal_node,
        create_leaf_node,
    )


def test_basic_workflow():
    """Minimal end-to-end: load data → evolve → predict."""
    from ga_trees import GAConfig, GAEngine, FitnessCalculator, Mutation, TreeInitializer
    from ga_trees.data import DatasetLoader
    from ga_trees.fitness import TreePredictor

    # Load data
    loader = DatasetLoader()
    data = loader.load_dataset("iris", test_size=0.2, random_state=42)
    X_train, y_train = data["X_train"], data["y_train"]
    X_test, y_test = data["X_test"], data["y_test"]

    n_features = X_train.shape[1]
    n_classes = len(np.unique(y_train))

    # Setup GA (tiny population / generations so the test is fast)
    config = GAConfig(population_size=10, n_generations=5)
    fitness_calc = FitnessCalculator()
    initializer = TreeInitializer(
        n_features, n_classes, max_depth=4, min_samples_split=5, min_samples_leaf=2
    )
    feature_ranges = {
        i: (float(X_train[:, i].min()), float(X_train[:, i].max()))
        for i in range(n_features)
    }
    mutation = Mutation(n_features, feature_ranges)
    engine = GAEngine(config, initializer, fitness_calc.calculate_fitness, mutation)

    best = engine.evolve(X_train, y_train, verbose=False)

    # Basic sanity checks on the returned tree
    assert best is not None, "evolve() returned None"
    assert best.fitness_ > 0, f"fitness_ should be positive, got {best.fitness_}"
    assert best.get_depth() <= 4, f"depth {best.get_depth()} exceeds max_depth=4"
    assert best.get_num_nodes() >= 1

    # Prediction works and returns an array of the right shape
    predictor = TreePredictor()
    y_pred = predictor.predict(best, X_test)
    assert y_pred.shape == y_test.shape, (
        f"Prediction shape {y_pred.shape} doesn't match target shape {y_test.shape}"
    )


def test_fitness_calculator_standalone():
    """FitnessCalculator runs without a full GA loop."""
    from ga_trees.fitness import FitnessCalculator
    from ga_trees.genotype import TreeGenotype, create_internal_node, create_leaf_node

    left = create_leaf_node(prediction=0, depth=1)
    right = create_leaf_node(prediction=1, depth=1)
    root = create_internal_node(0, 0.5, left, right, depth=0)
    tree = TreeGenotype(root=root, n_features=4, n_classes=2)

    X = np.array([[0.2, 0.5, 0.5, 0.5], [0.8, 0.5, 0.5, 0.5]])
    y = np.array([0, 1])

    fc = FitnessCalculator()
    fitness = fc.calculate_fitness(tree, X, y)

    assert 0.0 <= fitness <= 1.0
    assert hasattr(tree, "accuracy_")
    assert hasattr(tree, "interpretability_")


def test_dataset_loader_returns_expected_keys():
    """DatasetLoader must return all required keys for the GA pipeline."""
    from ga_trees.data import DatasetLoader

    loader = DatasetLoader()
    data = loader.load_dataset("iris", test_size=0.2, random_state=0)

    required_keys = {"X_train", "X_test", "y_train", "y_test", "metadata"}
    assert required_keys.issubset(data.keys()), (
        f"Missing keys: {required_keys - data.keys()}"
    )
    assert data["X_train"].shape[1] == 4
    assert len(np.unique(data["y_train"])) == 3


@pytest.mark.slow
def test_full_ga_run_with_history():
    """Full GA run (larger population) records history that improves over time."""
    from ga_trees import GAConfig, GAEngine, FitnessCalculator, Mutation, TreeInitializer
    from ga_trees.data import DatasetLoader

    loader = DatasetLoader()
    data = loader.load_dataset("iris", test_size=0.3, random_state=42)
    X_train, y_train = data["X_train"], data["y_train"]

    n_features = X_train.shape[1]
    n_classes = len(np.unique(y_train))
    feature_ranges = {
        i: (float(X_train[:, i].min()), float(X_train[:, i].max()))
        for i in range(n_features)
    }

    config = GAConfig(population_size=30, n_generations=20)
    fitness_calc = FitnessCalculator()
    initializer = TreeInitializer(n_features, n_classes, max_depth=5, min_samples_split=5, min_samples_leaf=2)
    mutation = Mutation(n_features, feature_ranges)
    engine = GAEngine(config, initializer, fitness_calc.calculate_fitness, mutation)

    engine.evolve(X_train, y_train, verbose=False)
    history = engine.get_history()

    assert "best_fitness" in history
    assert len(history["best_fitness"]) == 20

    first_half_avg = np.mean(history["best_fitness"][:10])
    second_half_avg = np.mean(history["best_fitness"][10:])
    assert second_half_avg >= first_half_avg, (
        "Fitness should not decrease over generations "
        f"(first half avg={first_half_avg:.4f}, second half avg={second_half_avg:.4f})"
    )
