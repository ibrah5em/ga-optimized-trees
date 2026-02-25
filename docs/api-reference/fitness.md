# Fitness Calculator API Reference

Complete API documentation for fitness evaluation.

## Module: `ga_trees.fitness.calculator`

### TreePredictor

Make predictions with tree genotypes.

Uses vectorized batch traversal (iterative, stack-based) for performance
and to avoid recursion depth issues on deep trees.

#### Static Methods

##### `predict(tree, X)`

Predict labels for input data.

**Parameters:**

- `tree` (TreeGenotype): Tree to use for prediction
- `X` (np.ndarray): Feature matrix of shape (n_samples, n_features)

**Returns:**

- `np.ndarray`: Predicted labels of shape (n_samples,)

**Raises:**

- `ValueError`: If `X` is not 2-D or has fewer features than the tree expects

**Example:**

```python
from ga_trees.fitness.calculator import TreePredictor

predictor = TreePredictor()
y_pred = predictor.predict(best_tree, X_test)
```

##### `fit_leaf_predictions(tree, X, y)`

Update leaf predictions based on training data.

**Parameters:**

- `tree` (TreeGenotype): Tree to update
- `X` (np.ndarray): Training features
- `y` (np.ndarray): Training labels

**Side Effects:**

- Updates `node.prediction` for all leaf nodes
- Classification: Sets to most common class in that leaf
- Regression: Sets to mean value in that leaf
- Unreachable leaves receive the dataset-global prior (majority class or mean) instead of a default 0

**Example:**

```python
# Fit predictions before evaluating
predictor.fit_leaf_predictions(tree, X_train, y_train)
y_pred = predictor.predict(tree, X_test)
```

______________________________________________________________________

### InterpretabilityCalculator

Calculate interpretability metrics.

#### Static Methods

##### `calculate_composite_score(tree, weights)`

Calculate composite interpretability score.

**Parameters:**

- `tree` (TreeGenotype): Tree to evaluate
- `weights` (dict): Component weights with keys:
  - `node_complexity`: Weight for tree size penalty
  - `feature_coherence`: Weight for feature reuse
  - `tree_balance`: Weight for balance metric
  - `semantic_coherence`: Weight for feature depth consistency

**Returns:**

- `float`: Interpretability score in \[0, 1\] (higher = more interpretable)

**Formula:**

```
score = Σ (weight_i × component_i)

Components:
  node_complexity:    1 - nodes(T) / max_nodes
                      where max_nodes = 2^(max_depth+1) - 1
  feature_coherence:  1 - (unique_features / total_features)
                      Returns 0.5 for leaf-only trees (no features used)
  tree_balance:       tree.get_tree_balance()
  semantic_coherence: mean consistency of feature depth positions
```

**Example:**

```python
from ga_trees.fitness.calculator import InterpretabilityCalculator

weights = {
    "node_complexity": 0.50,
    "feature_coherence": 0.10,
    "tree_balance": 0.10,
    "semantic_coherence": 0.30,
}

calc = InterpretabilityCalculator()
score = calc.calculate_composite_score(tree, weights)
print(f"Interpretability: {score:.4f}")
```

______________________________________________________________________

### FitnessCalculator

Main fitness calculator with multi-objective support.

#### Constructor

```python
FitnessCalculator(
    mode="weighted_sum",
    accuracy_weight=0.7,
    interpretability_weight=0.3,
    interpretability_weights=None,
    classification_metric="accuracy",
    regression_metric="neg_mse",
)
```

**Parameters:**

- `mode` (str): `'weighted_sum'` (returns scalar) or `'pareto'` (returns tuple)
- `accuracy_weight` (float): Weight for accuracy in \[0, 1\]
- `interpretability_weight` (float): Weight for interpretability in \[0, 1\]
- `interpretability_weights` (dict, optional): Sub-weights for interpretability components
- `classification_metric` (str): One of `'accuracy'`, `'f1_macro'`, `'f1_weighted'`, `'balanced_accuracy'`
- `regression_metric` (str): One of `'neg_mse'` (→ 1/(1+MSE)), `'r2'`

**Raises:**

- `ValueError`: If any parameter is out of its valid range

**Default Interpretability Weights:**

```python
{
    "node_complexity": 0.4,
    "feature_coherence": 0.3,
    "tree_balance": 0.2,
    "semantic_coherence": 0.1,
}
```

#### Methods

##### `calculate_fitness(tree, X, y, X_val=None, y_val=None)`

Calculate fitness score for tree.

**Parameters:**

- `tree` (TreeGenotype): Tree to evaluate
- `X` (np.ndarray): Training features (used to fit leaf predictions)
- `y` (np.ndarray): Training labels
- `X_val` (np.ndarray, optional): Validation features for generalization fitness
- `y_val` (np.ndarray, optional): Validation labels

**Returns:**

- `float`: Fitness score in \[0, 1\] (weighted_sum mode)
- `tuple[float, float]`: (accuracy, interpretability) (pareto mode)

**Side Effects:**

- Sets `tree.accuracy_`
- Sets `tree.interpretability_`
- Updates leaf predictions

**Example:**

```python
from ga_trees.fitness.calculator import FitnessCalculator

fitness_calc = FitnessCalculator(
    mode="weighted_sum",
    accuracy_weight=0.68,
    interpretability_weight=0.32,
    classification_metric="f1_weighted",  # better for imbalanced data
    interpretability_weights={
        "node_complexity": 0.50,
        "feature_coherence": 0.10,
        "tree_balance": 0.10,
        "semantic_coherence": 0.30,
    },
)

# With validation set (recommended)
fitness = fitness_calc.calculate_fitness(tree, X_train, y_train, X_val, y_val)

# Without validation set (evaluates on training data)
fitness = fitness_calc.calculate_fitness(tree, X_train, y_train)
```

______________________________________________________________________

## Custom Fitness Example

```python
from ga_trees.fitness.calculator import FitnessCalculator, TreePredictor
from sklearn.metrics import recall_score


class CustomFitness(FitnessCalculator):
    def calculate_fitness(self, tree, X, y, X_val=None, y_val=None):
        # Fit predictions
        self.predictor.fit_leaf_predictions(tree, X, y)

        # Evaluate on validation set if available
        X_eval = X_val if X_val is not None else X
        y_eval = y_val if y_val is not None else y

        y_pred = self.predictor.predict(tree, X_eval)

        # Custom metric (recall instead of accuracy)
        recall = recall_score(y_eval, y_pred, average="weighted")

        # Interpretability
        interp = self.interp_calc.calculate_composite_score(
            tree, self.interpretability_weights
        )

        # Store metrics
        tree.accuracy_ = recall
        tree.interpretability_ = interp

        # Weighted fitness
        fitness = self.accuracy_weight * recall + self.interpretability_weight * interp

        return fitness


# Use custom fitness
custom_fitness = CustomFitness(accuracy_weight=0.70, interpretability_weight=0.30)
fitness = custom_fitness.calculate_fitness(tree, X_train, y_train, X_val, y_val)
```
