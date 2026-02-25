# Quick Start Guide

Get started in 5 minutes.

## Installation

```bash
pip install -e .
```

## 1-Minute Demo

```bash
python scripts/train.py --dataset iris --generations 10 --population 30
```

## 5-Minute Tutorial

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from ga_trees.ga.engine import GAEngine, GAConfig, TreeInitializer, Mutation
from ga_trees.fitness.calculator import FitnessCalculator
import numpy as np

# Load and split data
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Setup
n_features, n_classes = X_train.shape[1], len(np.unique(y))
feature_ranges = {
    i: (X_train[:, i].min(), X_train[:, i].max()) for i in range(n_features)
}

# Configure
ga_config = GAConfig(population_size=30, n_generations=20)
initializer = TreeInitializer(
    n_features=n_features,
    n_classes=n_classes,
    max_depth=4,
    min_samples_split=5,
    min_samples_leaf=2,
)
fitness_calc = FitnessCalculator()
mutation = Mutation(n_features=n_features, feature_ranges=feature_ranges)

# Train
ga_engine = GAEngine(ga_config, initializer, fitness_calc.calculate_fitness, mutation)
best_tree = ga_engine.evolve(X_train, y_train)

# Evaluate
from ga_trees.fitness.calculator import TreePredictor

y_pred = TreePredictor().predict(best_tree, X_test)
from sklearn.metrics import accuracy_score

print(
    f"Accuracy: {accuracy_score(y_test, y_pred):.4f}, Nodes: {best_tree.get_num_nodes()}"
)
```

## Next Steps

- [Full Installation Guide](installation.md)
- [Configuration Guide](Configuration.md)
- [Training Tutorial](tutorial.md)
