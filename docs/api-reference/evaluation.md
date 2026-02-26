# Evaluation Tools API Reference

Comprehensive API documentation for evaluation, metrics, and analysis tools.

## Module: `ga_trees.evaluation.metrics`

### MetricsCalculator

Calculate comprehensive metrics for model evaluation.

#### Methods

##### `calculate_classification_metrics(y_true, y_pred, y_prob=None)`

Calculate classification metrics.

**Parameters:**

- `y_true` (np.ndarray): True labels
- `y_pred` (np.ndarray): Predicted labels
- `y_prob` (np.ndarray, optional): Prediction probabilities

**Returns:**

- `dict`: Dictionary with metrics:
  - `accuracy`: Overall accuracy
  - `precision_macro`: Macro-averaged precision
  - `precision_weighted`: Weighted precision
  - `recall_macro`: Macro-averaged recall
  - `recall_weighted`: Weighted recall
  - `f1_macro`: Macro F1-score
  - `f1_weighted`: Weighted F1-score
  - `confusion_matrix`: Confusion matrix
  - `roc_auc` (if binary): ROC-AUC score

**Example:**

```python
from ga_trees.evaluation.metrics import MetricsCalculator

calculator = MetricsCalculator()
metrics = calculator.calculate_classification_metrics(y_test, y_pred)

print(f"Accuracy: {metrics['accuracy']:.4f}")
print(f"F1 (weighted): {metrics['f1_weighted']:.4f}")
print(f"Confusion Matrix:\n{metrics['confusion_matrix']}")
```

##### `calculate_interpretability_metrics(tree)`

Calculate interpretability metrics for a tree.

**Parameters:**

- `tree` (TreeGenotype): Tree to evaluate

**Returns:**

- `dict`: Dictionary with:
  - `depth`: Tree depth
  - `num_nodes`: Number of nodes
  - `num_leaves`: Number of leaves
  - `features_used`: Number of features used
  - `tree_balance`: Balance score \[0, 1\]
  - `interpretability_score`: Overall score

**Example:**

```python
interp_metrics = calculator.calculate_interpretability_metrics(best_tree)
print(f"Depth: {interp_metrics['depth']}")
print(f"Nodes: {interp_metrics['num_nodes']}")
print(f"Balance: {interp_metrics['tree_balance']:.4f}")
```

##### `print_classification_report(y_true, y_pred, target_names=None)`

Print detailed classification report.

**Parameters:**

- `y_true` (np.ndarray): True labels
- `y_pred` (np.ndarray): Predicted labels
- `target_names` (list, optional): Class names

**Example:**

```python
calculator.print_classification_report(
    y_test, y_pred, target_names=["setosa", "versicolor", "virginica"]
)
```

______________________________________________________________________

## Module: `ga_trees.evaluation.feature_importance`

### FeatureImportanceAnalyzer

Analyze feature importance in evolved trees.

#### Static Methods

##### `calculate_feature_frequency(tree)`

Count feature usage frequency.

**Parameters:**

- `tree` (TreeGenotype): Tree to analyze

**Returns:**

- `dict`: Mapping feature_idx → count

**Example:**

```python
from ga_trees.evaluation.feature_importance import FeatureImportanceAnalyzer

analyzer = FeatureImportanceAnalyzer()
freq = analyzer.calculate_feature_frequency(best_tree)
print(f"Feature usage: {freq}")  # {0: 3, 1: 2, 2: 1}
```

##### `calculate_feature_depth_importance(tree)`

Calculate importance based on depth (higher = more important).

**Parameters:**

- `tree` (TreeGenotype): Tree to analyze

**Returns:**

- `dict`: Mapping feature_idx → importance score \[0, 1\]

**Formula:**

```
importance(feature) = Σ (1 / (depth + 1)) for all nodes using feature
normalized by total
```

**Example:**

```python
importance = analyzer.calculate_feature_depth_importance(best_tree)
print(f"Feature 0 importance: {importance[0]:.4f}")
```

##### `plot_feature_importance(importance, feature_names=None, save_path=None)`

Visualize feature importance.

**Parameters:**

- `importance` (dict): Feature importance scores
- `feature_names` (list, optional): Feature names
- `save_path` (str, optional): Path to save plot

**Example:**

```python
importance = analyzer.calculate_feature_depth_importance(best_tree)
analyzer.plot_feature_importance(
    importance,
    feature_names=["sepal_length", "sepal_width", "petal_length", "petal_width"],
    save_path="results/figures/feature_importance.png",
)
```

______________________________________________________________________

## Module: `ga_trees.evaluation.tree_visualizer`

### TreeVisualizer

Visualize decision trees with Graphviz.

**Requirements:** `pip install graphviz`

#### Static Methods

##### `tree_to_graphviz(tree, feature_names=None, class_names=None)`

Convert tree to Graphviz format.

**Parameters:**

- `tree` (TreeGenotype): Tree to visualize
- `feature_names` (list, optional): Feature names
- `class_names` (list, optional): Class names

**Returns:**

- `graphviz.Digraph`: Graphviz diagram object

**Example:**

```python
from ga_trees.evaluation.tree_visualizer import TreeVisualizer

feature_names = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
class_names = ["setosa", "versicolor", "virginica"]

dot = TreeVisualizer.tree_to_graphviz(
    best_tree, feature_names=feature_names, class_names=class_names
)

# Display in Jupyter
dot
```

##### `visualize_tree(tree, feature_names=None, class_names=None, save_path='results/figures/tree_viz')`

Visualize and save tree.

**Parameters:**

- `tree` (TreeGenotype): Tree to visualize
- `feature_names` (list, optional): Feature names
- `class_names` (list, optional): Class names
- `save_path` (str): Output path (without extension)

**Example:**

```python
TreeVisualizer.visualize_tree(
    best_tree,
    feature_names=feature_names,
    class_names=class_names,
    save_path="results/figures/my_tree",
)
# Creates: my_tree.png
```

______________________________________________________________________

## Module: `ga_trees.evaluation.explainability`

### TreeExplainer

Model explainability with LIME and SHAP.

**Optional Requirements:** `pip install shap lime`

#### Static Methods

##### `explain_with_shap(tree, X, feature_names=None)`

Use SHAP to explain tree predictions.

**Parameters:**

- `tree` (TreeGenotype): Tree to explain
- `X` (np.ndarray): Data to explain
- `feature_names` (list, optional): Feature names

**Example:**

```python
from ga_trees.evaluation.explainability import TreeExplainer

# SHAP explanation
TreeExplainer.explain_with_shap(
    best_tree,
    X_test,
    feature_names=["sepal_length", "sepal_width", "petal_length", "petal_width"],
)
# Displays SHAP summary plot
```

##### `explain_with_lime(tree, X, instance_idx, feature_names=None)`

Use LIME to explain single prediction.

**Parameters:**

- `tree` (TreeGenotype): Tree to explain
- `X` (np.ndarray): Full dataset
- `instance_idx` (int): Index of instance to explain
- `feature_names` (list, optional): Feature names

**Example:**

```python
# Explain prediction for instance 42
TreeExplainer.explain_with_lime(
    best_tree, X_test, instance_idx=42, feature_names=feature_names
)
```

______________________________________________________________________

## Complete Evaluation Pipeline

```python
from ga_trees.evaluation.metrics import MetricsCalculator
from ga_trees.evaluation.feature_importance import FeatureImportanceAnalyzer
from ga_trees.evaluation.tree_visualizer import TreeVisualizer
from ga_trees.fitness.calculator import TreePredictor
import numpy as np

# After training
best_tree = ga_engine.evolve(X_train, y_train)

# 1. Make predictions
predictor = TreePredictor()
y_pred = predictor.predict(best_tree, X_test)

# 2. Calculate metrics
calculator = MetricsCalculator()
metrics = calculator.calculate_classification_metrics(y_test, y_pred)
interp_metrics = calculator.calculate_interpretability_metrics(best_tree)

print("=" * 60)
print("EVALUATION RESULTS")
print("=" * 60)
print(f"\nAccuracy: {metrics['accuracy']:.4f}")
print(f"F1-Score: {metrics['f1_weighted']:.4f}")
print(f"Precision: {metrics['precision_weighted']:.4f}")
print(f"Recall: {metrics['recall_weighted']:.4f}")

print(f"\nTree Metrics:")
print(f"  Depth: {interp_metrics['depth']}")
print(f"  Nodes: {interp_metrics['num_nodes']}")
print(f"  Leaves: {interp_metrics['num_leaves']}")
print(f"  Features Used: {interp_metrics['features_used']}")
print(f"  Balance: {interp_metrics['tree_balance']:.4f}")

# 3. Feature importance
analyzer = FeatureImportanceAnalyzer()
importance = analyzer.calculate_feature_depth_importance(best_tree)
analyzer.plot_feature_importance(
    importance,
    feature_names=["feature_" + str(i) for i in range(X_train.shape[1])],
    save_path="results/figures/feature_importance.png",
)

# 4. Visualize tree
TreeVisualizer.visualize_tree(
    best_tree,
    feature_names=["feature_" + str(i) for i in range(X_train.shape[1])],
    save_path="results/figures/tree_structure",
)

# 5. Print classification report
calculator.print_classification_report(y_test, y_pred)
```
