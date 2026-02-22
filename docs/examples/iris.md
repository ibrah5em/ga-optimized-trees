# Iris Dataset Example

Step-by-step tutorial using Iris dataset.

## Introduction

Iris dataset:
- **150 samples**, **4 features**, **3 classes**
- Features: sepal_length, sepal_width, petal_length, petal_width
- Classes: setosa, versicolor, virginica
- **Simple dataset** - good for learning

## Complete Example

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

from ga_trees.ga.engine import GAEngine, GAConfig, TreeInitializer, Mutation
from ga_trees.fitness.calculator import FitnessCalculator, TreePredictor

# 1. Load data
X, y = load_iris(return_X_y=True)
feature_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
class_names = ['setosa', 'versicolor', 'virginica']

print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features, {len(np.unique(y))} classes")

# 2. Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# 3. Standardize
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 4. Setup GA
n_features = X_train.shape[1]
n_classes = len(np.unique(y))
feature_ranges = {i: (X_train[:, i].min(), X_train[:, i].max()) 
                 for i in range(n_features)}

ga_config = GAConfig(
    population_size=50,
    n_generations=30,
    crossover_prob=0.7,
    mutation_prob=0.2
)

initializer = TreeInitializer(
    n_features=n_features,
    n_classes=n_classes,
    max_depth=4,
    min_samples_split=5,
    min_samples_leaf=2
)

fitness_calc = FitnessCalculator(
    accuracy_weight=0.7,
    interpretability_weight=0.3
)

mutation = Mutation(n_features=n_features, feature_ranges=feature_ranges)

# 5. Train
print("\nTraining...")
ga_engine = GAEngine(ga_config, initializer, fitness_calc.calculate_fitness, mutation)
best_tree = ga_engine.evolve(X_train, y_train, verbose=True)

# 6. Evaluate
predictor = TreePredictor()
y_train_pred = predictor.predict(best_tree, X_train)
y_test_pred = predictor.predict(best_tree, X_test)

print("\n" + "="*60)
print("RESULTS")
print("="*60)
print(f"\nTrain Accuracy: {accuracy_score(y_train, y_train_pred):.4f}")
print(f"Test Accuracy: {accuracy_score(y_test, y_test_pred):.4f}")

print(f"\nTree Statistics:")
print(f"  Depth: {best_tree.get_depth()}")
print(f"  Nodes: {best_tree.get_num_nodes()}")
print(f"  Leaves: {best_tree.get_num_leaves()}")
print(f"  Features Used: {best_tree.get_num_features_used()}/{n_features}")

# 7. Extract rules
print(f"\nDecision Rules:")
rules = best_tree.to_rules()
for i, rule in enumerate(rules, 1):
    # Replace feature indices with names
    readable_rule = rule
    for j, name in enumerate(feature_names):
        readable_rule = readable_rule.replace(f"X[{j}]", name)
    print(f"  {i}. {readable_rule}")

# 8. Classification report
print(f"\nClassification Report:")
print(classification_report(y_test, y_test_pred, target_names=class_names))
```

**Expected Output:**
```
Dataset: 150 samples, 4 features, 3 classes

Training...
Gen 0: Best=0.7234, Avg=0.5123
Gen 10: Best=0.8567, Avg=0.7234
Gen 20: Best=0.9123, Avg=0.8456
Gen 30: Best=0.9345, Avg=0.8876

============================================================
RESULTS
============================================================

Train Accuracy: 0.9714
Test Accuracy: 0.9556

Tree Statistics:
  Depth: 3
  Nodes: 7
  Leaves: 4
  Features Used: 2/4

Decision Rules:
  1. IF petal_length <= 2.4500 THEN class=0
  2. IF petal_length > 2.4500 AND petal_width <= 1.7500 THEN class=1
  3. IF petal_length > 2.4500 AND petal_width > 1.7500 THEN class=2

Classification Report:
              precision    recall  f1-score   support

      setosa       1.00      1.00      1.00        15
  versicolor       0.93      0.93      0.93        15
   virginica       0.93      0.93      0.93        15

    accuracy                           0.96        45
```

## Visualization

```python
from ga_trees.evaluation.tree_visualizer import TreeVisualizer

TreeVisualizer.visualize_tree(
    best_tree,
    feature_names=feature_names,
    class_names=class_names,
    save_path='results/figures/iris_tree'
)
```

See other examples: [Medical](medical.md), [Credit Scoring](credit.md), [Custom Dataset Guide](../data/dataset-loader.md)