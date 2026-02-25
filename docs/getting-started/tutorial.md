# Complete Tutorial

Step-by-step guide from setup to evaluation.

## Tutorial Overview

This tutorial covers:

1. Installation and setup
1. Loading and preprocessing data
1. Configuring the GA
1. Training a model
1. Evaluating results
1. Visualizing the tree
1. Comparing with baselines

## Step 1: Installation

```bash
git clone https://github.com/ibrah5em/ga-optimized-trees.git
cd ga-optimized-trees
pip install -e .
```

## Step 2: Load Data

```python
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load
X, y = load_breast_cancer(return_X_y=True)
print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Standardize
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```

## Step 3: Configure GA

```python
import numpy as np
from ga_trees.ga.engine import GAConfig, TreeInitializer, Mutation
from ga_trees.fitness.calculator import FitnessCalculator

# Setup parameters
n_features = X_train.shape[1]
n_classes = len(np.unique(y))
feature_ranges = {
    i: (X_train[:, i].min(), X_train[:, i].max()) for i in range(n_features)
}

# GA configuration
ga_config = GAConfig(
    population_size=80, n_generations=40, crossover_prob=0.7, mutation_prob=0.2
)

# Tree constraints
initializer = TreeInitializer(
    n_features=n_features,
    n_classes=n_classes,
    max_depth=6,
    min_samples_split=10,
    min_samples_leaf=5,
)

# Fitness function
fitness_calc = FitnessCalculator(accuracy_weight=0.7, interpretability_weight=0.3)

# Mutation operator
mutation = Mutation(n_features=n_features, feature_ranges=feature_ranges)
```

## Step 4: Train Model

```python
from ga_trees.ga.engine import GAEngine

ga_engine = GAEngine(
    config=ga_config,
    initializer=initializer,
    fitness_function=fitness_calc.calculate_fitness,
    mutation=mutation,
)

print("Training...")
best_tree = ga_engine.evolve(X_train, y_train, verbose=True)
print(f"Training complete! Best fitness: {best_tree.fitness_:.4f}")
```

## Step 5: Evaluate

```python
from ga_trees.fitness.calculator import TreePredictor
from sklearn.metrics import accuracy_score, classification_report

predictor = TreePredictor()
y_pred = predictor.predict(best_tree, X_test)

print("\n" + "=" * 60)
print("EVALUATION RESULTS")
print("=" * 60)
print(f"\nTest Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"\nTree Statistics:")
print(f"  Nodes: {best_tree.get_num_nodes()}")
print(f"  Depth: {best_tree.get_depth()}")
print(f"  Leaves: {best_tree.get_num_leaves()}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))
```

## Step 6: Visualize

```python
from ga_trees.evaluation.tree_visualizer import TreeVisualizer

TreeVisualizer.visualize_tree(
    best_tree,
    feature_names=load_breast_cancer().feature_names,
    class_names=["benign", "malignant"],
    save_path="results/figures/breast_cancer_tree",
)
print("Tree visualization saved to results/figures/breast_cancer_tree.png")
```

## Step 7: Compare with CART

```python
from sklearn.tree import DecisionTreeClassifier

cart = DecisionTreeClassifier(max_depth=6, random_state=42)
cart.fit(X_train, y_train)
y_pred_cart = cart.predict(X_test)

print("\n" + "=" * 60)
print("COMPARISON: GA vs CART")
print("=" * 60)
print(f"\nGA Accuracy:   {accuracy_score(y_test, y_pred):.4f}")
print(f"CART Accuracy: {accuracy_score(y_test, y_pred_cart):.4f}")
print(f"\nGA Nodes:   {best_tree.get_num_nodes()}")
print(f"CART Nodes: {cart.tree_.node_count}")
print(
    f"\nSize Reduction: {(1 - best_tree.get_num_nodes()/cart.tree_.node_count)*100:.1f}%"
)
```

## Next Steps

- [Run Full Experiments](../user-guides/experiments.md)
- [Hyperparameter Tuning](../user-guides/hyperparameter-tuning.md)
- [Custom Fitness Functions](../advanced/custom-fitness.md)
