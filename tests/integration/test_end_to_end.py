"""End-to-end integration tests."""

import numpy as np
import pytest
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from ga_trees.fitness.calculator import FitnessCalculator, TreePredictor
from ga_trees.ga.engine import GAConfig, GAEngine, Mutation, TreeInitializer


@pytest.fixture
def iris_data():
    """Load iris dataset."""
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    return X_train, X_test, y_train, y_test


def test_full_training_pipeline(iris_data):
    """Test complete training pipeline."""
    X_train, X_test, y_train, y_test = iris_data

    # Setup
    n_features = X_train.shape[1]
    n_classes = len(np.unique(y_train))
    feature_ranges = {i: (X_train[:, i].min(), X_train[:, i].max()) for i in range(n_features)}

    ga_config = GAConfig(
        population_size=20,
        n_generations=10,
        crossover_prob=0.7,
        mutation_prob=0.2,
        tournament_size=3,
    )

    initializer = TreeInitializer(
        n_features=n_features,
        n_classes=n_classes,
        max_depth=4,
        min_samples_split=5,
        min_samples_leaf=2,
    )

    fitness_calc = FitnessCalculator(
        mode="weighted_sum", accuracy_weight=0.8, interpretability_weight=0.2
    )

    mutation = Mutation(n_features=n_features, feature_ranges=feature_ranges)

    # Create and run GA
    ga_engine = GAEngine(
        config=ga_config,
        initializer=initializer,
        fitness_function=fitness_calc.calculate_fitness,
        mutation=mutation,
    )

    best_tree = ga_engine.evolve(X_train, y_train, verbose=False)

    # Verify results
    assert best_tree is not None
    assert best_tree.fitness_ > 0

    # Test prediction
    predictor = TreePredictor()
    y_pred = predictor.predict(best_tree, X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # Should get reasonable accuracy on iris
    assert accuracy > 0.6, f"Accuracy too low: {accuracy}"

    # Tree should be interpretable
    assert best_tree.get_depth() <= 4
    assert best_tree.get_num_nodes() < 31  # Max for depth 4


def test_fitness_improves(iris_data):
    """Test that fitness improves over generations."""
    X_train, _, y_train, _ = iris_data

    n_features = X_train.shape[1]
    n_classes = len(np.unique(y_train))
    feature_ranges = {i: (X_train[:, i].min(), X_train[:, i].max()) for i in range(n_features)}

    ga_config = GAConfig(population_size=30, n_generations=20)
    initializer = TreeInitializer(
        n_features=n_features,
        n_classes=n_classes,
        max_depth=5,
        min_samples_split=5,
        min_samples_leaf=2,
    )
    fitness_calc = FitnessCalculator()
    mutation = Mutation(n_features=n_features, feature_ranges=feature_ranges)

    ga_engine = GAEngine(ga_config, initializer, fitness_calc.calculate_fitness, mutation)

    ga_engine.evolve(X_train, y_train, verbose=False)

    history = ga_engine.get_history()

    # Fitness should generally improve
    first_10_avg = np.mean(history["best_fitness"][:10])
    last_10_avg = np.mean(history["best_fitness"][-10:])

    assert last_10_avg >= first_10_avg, "Fitness didn't improve"
