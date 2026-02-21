# Baseline Model Comparisons

This guide covers how to compare GA-optimized trees against traditional machine learning baselines.

## Overview

The framework provides comprehensive baseline comparisons to validate that GA-optimized trees offer competitive performance while maintaining interpretability. Supported baselines include:

- **CART** - Standard decision tree (scikit-learn)
- **Pruned CART** - Cost-complexity pruned CART
- **Random Forest** - Ensemble baseline
- **XGBoost** - Gradient boosting baseline (optional)

## Quick Comparison

```bash
# Run full baseline comparison
python scripts/experiment.py --config configs/paper.yaml

# Output includes statistical tests comparing GA vs baselines
```

## Using Baseline Models

### 1. CART Baseline

```python
from ga_trees.baselines.baseline_models import CARTBaseline
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

# Load data
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Train CART
cart = CARTBaseline(max_depth=5, min_samples_split=10)
cart.fit(X_train, y_train)

# Predict
y_pred = cart.predict(X_test)

# Get metrics
metrics = cart.get_metrics()
print(f"Depth: {metrics['depth']}")
print(f"Nodes: {metrics['num_nodes']}")
print(f"Leaves: {metrics['num_leaves']}")
```

### 2. Pruned CART

```python
from ga_trees.baselines.baseline_models import PrunedCARTBaseline

# Pruned CART uses cost-complexity pruning
pruned = PrunedCARTBaseline(max_depth=10)
pruned.fit(X_train, y_train)
y_pred = pruned.predict(X_test)

# Typically produces smaller trees than unpruned CART
print(f"Pruned nodes: {pruned.get_num_nodes()}")
```

### 3. Random Forest

```python
from ga_trees.baselines.baseline_models import RandomForestBaseline

# Random Forest ensemble
rf = RandomForestBaseline(n_estimators=100, max_depth=5)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

# Note: RF is not interpretable but provides accuracy upper bound
```

### 4. XGBoost (Optional)

```python
from ga_trees.baselines.baseline_models import XGBoostBaseline

# Requires: pip install xgboost
xgb = XGBoostBaseline(max_depth=6, n_estimators=100)
xgb.fit(X_train, y_train)
y_pred = xgb.predict(X_test)
```

## Comprehensive Comparison

### Statistical Testing

```python
import numpy as np
from scipy import stats
from sklearn.model_selection import StratifiedKFold

# Setup
n_folds = 20
skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

ga_scores = []
cart_scores = []

for train_idx, test_idx in skf.split(X, y):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    # Train GA
    ga_tree = train_ga(X_train, y_train)  # Your GA training
    ga_scores.append(evaluate(ga_tree, X_test, y_test))
    
    # Train CART
    cart = CARTBaseline(max_depth=5)
    cart.fit(X_train, y_train)
    cart_scores.append(evaluate_cart(cart, X_test, y_test))

# Statistical test
t_stat, p_value = stats.ttest_rel(ga_scores, cart_scores)
print(f"p-value: {p_value:.4f}")

if p_value > 0.05:
    print("No significant difference (statistically equivalent)")
else:
    print("Significant difference detected")
```

## Comparison Metrics

### 1. Accuracy Metrics

```python
from sklearn.metrics import accuracy_score, f1_score, classification_report

# Compare multiple metrics
metrics = {
    'GA': {
        'accuracy': accuracy_score(y_test, y_pred_ga),
        'f1': f1_score(y_test, y_pred_ga, average='weighted')
    },
    'CART': {
        'accuracy': accuracy_score(y_test, y_pred_cart),
        'f1': f1_score(y_test, y_pred_cart, average='weighted')
    }
}

print(classification_report(y_test, y_pred_ga))
```

### 2. Interpretability Metrics

```python
# Compare tree complexity
comparison = {
    'Model': ['GA', 'CART', 'RF'],
    'Nodes': [
        ga_tree.get_num_nodes(),
        cart.get_num_nodes(),
        'N/A'  # Ensemble
    ],
    'Depth': [
        ga_tree.get_depth(),
        cart.get_depth(),
        'N/A'
    ],
    'Interpretability Score': [
        ga_tree.interpretability_,
        calculate_cart_interpretability(cart),
        0.0  # Not interpretable
    ]
}
```

## Results Visualization

### Accuracy Comparison

```python
import matplotlib.pyplot as plt
import pandas as pd

results = pd.DataFrame({
    'Model': ['GA', 'CART', 'RF', 'XGBoost'],
    'Accuracy': [0.945, 0.924, 0.953, 0.958],
    'Std': [0.081, 0.104, 0.034, 0.028]
})

plt.figure(figsize=(10, 6))
plt.bar(results['Model'], results['Accuracy'], yerr=results['Std'])
plt.ylabel('Accuracy')
plt.title('Model Accuracy Comparison')
plt.ylim([0.85, 1.0])
plt.grid(axis='y', alpha=0.3)
plt.savefig('results/figures/accuracy_comparison.png')
```

### Size Comparison

```python
sizes = {
    'GA': 7.4,
    'CART': 16.4,
    'Pruned CART': 12.8
}

plt.figure(figsize=(10, 6))
plt.bar(sizes.keys(), sizes.values())
plt.ylabel('Number of Nodes')
plt.title('Tree Size Comparison')
plt.savefig('results/figures/size_comparison.png')
```

## Benchmark Results

### Target Results (configs/paper.yaml)

| Dataset | GA Acc | CART Acc | p-value | GA Nodes | CART Nodes | Reduction |
|---------|--------|----------|---------|----------|------------|-----------|
| Iris | 94.55% | 92.41% | 0.186 | 7.4 | 16.4 | 55% |
| Wine | 88.19% | 87.22% | 0.683 | 10.7 | 20.7 | 48% |
| Breast Cancer | 91.05% | 91.57% | 0.640 | 6.5 | 35.5 | 82% |

**Key Findings:**
- All p-values > 0.05 → Statistical equivalence ✓
- GA produces 46-82% smaller trees
- Minimal accuracy loss for significant size reduction

## Custom Baseline Addition

### Create Custom Baseline

```python
from ga_trees.baselines.baseline_models import BaselineModel

class MyCustomBaseline(BaselineModel):
    def __init__(self, **kwargs):
        super().__init__("MyCustomModel")
        self.model = YourModelClass(**kwargs)
    
    def fit(self, X, y):
        self.model.fit(X, y)
        return self
    
    def predict(self, X):
        return self.model.predict(X)
    
    def get_depth(self):
        # Implement if applicable
        return self.model.get_depth()
    
    def get_num_nodes(self):
        # Implement if applicable
        return self.model.get_num_nodes()
```

### Add to Experiment

```python
# In your experiment script
from ga_trees.baselines.baseline_models import CARTBaseline, RandomForestBaseline
from my_module import MyCustomBaseline

baselines = {
    'CART': CARTBaseline(max_depth=5),
    'RF': RandomForestBaseline(n_estimators=100),
    'Custom': MyCustomBaseline(param=value)
}

for name, baseline in baselines.items():
    baseline.fit(X_train, y_train)
    y_pred = baseline.predict(X_test)
    print(f"{name}: {accuracy_score(y_test, y_pred):.4f}")
```

## Best Practices

### 1. Fair Comparison

```python
# Use same constraints for all models
max_depth = 6
min_samples_split = 8
min_samples_leaf = 3

# CART
cart = CARTBaseline(
    max_depth=max_depth,
    min_samples_split=min_samples_split,
    min_samples_leaf=min_samples_leaf
)

# GA
ga_config = {
    'tree': {
        'max_depth': max_depth,
        'min_samples_split': min_samples_split,
        'min_samples_leaf': min_samples_leaf
    }
}
```

### 2. Sufficient Cross-Validation

```python
# Use 20-fold CV for statistical rigor
n_folds = 20
skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
```

### 3. Multiple Metrics

```python
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

metrics = {
    'accuracy': accuracy_score,
    'precision': lambda y, p: precision_score(y, p, average='weighted'),
    'recall': lambda y, p: recall_score(y, p, average='weighted'),
    'f1': lambda y, p: f1_score(y, p, average='weighted')
}

for metric_name, metric_fn in metrics.items():
    ga_score = metric_fn(y_test, y_pred_ga)
    cart_score = metric_fn(y_test, y_pred_cart)
    print(f"{metric_name}: GA={ga_score:.4f}, CART={cart_score:.4f}")
```

## Troubleshooting

### Issue: CART outperforms GA significantly

**Solutions:**
1. Increase accuracy weight:
   ```yaml
   fitness:
     weights:
       accuracy: 0.80  # Increase from 0.68
       interpretability: 0.20
   ```

2. Increase population/generations:
   ```yaml
   ga:
     population_size: 120
     n_generations: 60
   ```

3. Run hyperparameter optimization:
   ```bash
   python scripts/hyperopt_with_optuna.py --dataset your_dataset
   ```

### Issue: Statistical tests show significance when not expected

**Check:**
- Sample size (use more CV folds)
- Random seed consistency
- Data preprocessing consistency
- Evaluation metric calculation

## Next Steps

- [Hyperparameter Tuning](../user-guides/hyperparameter-tuning.md)
- [Statistical Tests](statistical-tests.md)
- [Visualization](../user-guides/visualization.md)