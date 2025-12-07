# Fitness Calculator API Reference

Complete API documentation for fitness evaluation.

## Module: `ga_trees.fitness.calculator`

### TreePredictor

Make predictions with tree genotypes.

#### Static Methods

##### `predict(tree, X)`

Predict labels for input data.

**Parameters:**
- `tree` (TreeGenotype): Tree to use for prediction
- `X` (np.ndarray): Feature matrix of shape (n_samples, n_features)

**Returns:**
- `np.ndarray`: Predicted labels of shape (n_samples,)

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
- Classification: Sets to most common class
- Regression: Sets to mean value

**Example:**
```python
# Fit predictions before evaluating
predictor.fit_leaf_predictions(tree, X_train, y_train)
y_pred = predictor.predict(tree, X_test)
```

---

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
  - `semantic_coherence`: Weight for prediction consistency

**Returns:**
- `float`: Interpretability score in [0, 1] (higher = more interpretable)

**Formula:**
```
score = Σ (weight_i × component_i)

Components:
  node_complexity: 1 / (1 + nodes / 127)
  feature_coherence: 1 - (unique_features / internal_nodes)
  tree_balance: tree.get_tree_balance()
  semantic_coherence: 1 - (prediction_entropy / max_entropy)
```

**Example:**
```python
from ga_trees.fitness.calculator import InterpretabilityCalculator

weights = {
    'node_complexity': 0.50,
    'feature_coherence': 0.10,
    'tree_balance': 0.10,
    'semantic_coherence': 0.30
}

calc = InterpretabilityCalculator()
score = calc.calculate_composite_score(tree, weights)
print(f"Interpretability: {score:.4f}")
```

---

### FitnessCalculator

Main fitness calculator with multi-objective support.

#### Constructor

```python
FitnessCalculator(
    mode='weighted_sum',
    accuracy_weight=0.7,
    interpretability_weight=0.3,
    interpretability_weights=None
)
```

**Parameters:**
- `mode` (str): 'weighted_sum' or 'pareto'
- `accuracy_weight` (float): Weight for accuracy [0, 1]
- `interpretability_weight` (float): Weight for interpretability [0, 1]
- `interpretability_weights` (dict, optional): Sub-weights for interpretability components

**Default Interpretability Weights:**
```python
{
    'node_complexity': 0.4,
    'feature_coherence': 0.3,
    'tree_balance': 0.2,
    'semantic_coherence': 0.1
}
```

#### Methods

##### `calculate_fitness(tree, X, y)`

Calculate fitness score for tree.

**Parameters:**
- `tree` (TreeGenotype): Tree to evaluate
- `X` (np.ndarray): Feature matrix
- `y` (np.ndarray): Labels

**Returns:**
- `float`: Fitness score in [0, 1]

**Side Effects:**
- Sets `tree.fitness_`
- Sets `tree.accuracy_`
- Sets `tree.interpretability_`
- Updates leaf predictions

**Example:**
```python
from ga_trees.fitness.calculator import FitnessCalculator

fitness_calc = FitnessCalculator(
    mode='weighted_sum',
    accuracy_weight=0.68,
    interpretability_weight=0.32,
    interpretability_weights={
        'node_complexity': 0.50,
        'feature_coherence': 0.10,
        'tree_balance': 0.10,
        'semantic_coherence': 0.30
    }
)

fitness = fitness_calc.calculate_fitness(tree, X_train, y_train)
print(f"Fitness: {fitness:.4f}")
print(f"Accuracy: {tree.accuracy_:.4f}")
print(f"Interpretability: {tree.interpretability_:.4f}")
```

---

## Custom Fitness Example

```python
from ga_trees.fitness.calculator import FitnessCalculator, TreePredictor
from sklearn.metrics import recall_score

class CustomFitness(FitnessCalculator):
    def calculate_fitness(self, tree, X, y):
        # Fit predictions
        self.predictor.fit_leaf_predictions(tree, X, y)
        y_pred = self.predictor.predict(tree, X)
        
        # Custom metric (recall instead of accuracy)
        recall = recall_score(y, y_pred, average='weighted')
        
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
fitness = custom_fitness.calculate_fitness(tree, X, y)
```