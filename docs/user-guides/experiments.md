# Running Experiments

Comprehensive guide to running benchmark experiments and comparing GA-optimized trees against baseline models.

## Overview

The experiments framework provides:

- **Automated benchmarking** across multiple datasets
- **Statistical rigor** with k-fold cross-validation
- **Baseline comparisons** (CART, Random Forest, XGBoost)
- **Statistical significance testing**
- **Reproducible results** with configurable parameters

## Quick Start

```bash
# Run experiments with default config (~17 minutes)
python scripts/experiment.py

# Run with custom config
python scripts/experiment.py --config configs/custom.yaml

# If using local dataset files, you can specify the label column by name or index
python scripts/experiment.py --datasets data/my_data.csv --label-column target
python scripts/experiment.py --datasets data/my_data.csv --label-column 4

# Run experiments using optimized config
python scripts/experiment.py --config configs/optimized.yaml
```

## Experiment Workflow

### 1. Basic Experiment Script

The `scripts/experiment.py` conducts a complete benchmark study:

```bash
python scripts/experiment.py --config configs/custom.yaml
```

**What it does:**

1. Loads datasets (Iris, Wine, Breast Cancer)
2. Runs 20-fold cross-validation for each:
    - GA-Optimized Trees
    - CART baseline
    - Random Forest baseline
3. Computes statistical tests (t-tests, Cohen's d)
4. Saves results to CSV and YAML

**Expected runtime:**

- Iris: ~5 minutes
- Wine: ~6 minutes
- Breast Cancer: ~6 minutes
- **Total: ~17 minutes**

### 2. Configuration-Driven Experiments

Create custom experiment config (`experiment_config.yaml`):

```yaml
# GA Configuration
ga:
  population_size: 80
  n_generations: 40
  crossover_prob: 0.72
  mutation_prob: 0.18
  tournament_size: 4
  elitism_ratio: 0.12
  
  mutation_types:
    threshold_perturbation: 0.45
    feature_replacement: 0.25
    prune_subtree: 0.25
    expand_leaf: 0.05

# Tree Constraints
tree:
  max_depth: 6
  min_samples_split: 8
  min_samples_leaf: 3

# Fitness Function
fitness:
  mode: weighted_sum
  weights:
    accuracy: 0.68
    interpretability: 0.32
  
  interpretability_weights:
    node_complexity: 0.50
    feature_coherence: 0.10
    tree_balance: 0.10
    semantic_coherence: 0.30

# Experiment Setup
experiment:
  datasets:
    - iris
    - wine
    - breast_cancer
  cv_folds: 20
  random_state: 42
```

Run with config:

```bash
python scripts/experiment.py --config experiment_config.yaml
```

### 3. Manual Experiment Loop

For custom experiment logic:

```python
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris, load_wine, load_breast_cancer
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from scipy import stats

from ga_trees.ga.engine import GAEngine, GAConfig, TreeInitializer, Mutation
from ga_trees.fitness.calculator import FitnessCalculator, TreePredictor

# Load dataset
X, y = load_breast_cancer(return_X_y=True)
dataset_name = "Breast Cancer"

# Cross-validation setup
n_folds = 20
skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

# Storage for results
ga_results = {'test_acc': [], 'test_f1': [], 'nodes': [], 'depth': []}
cart_results = {'test_acc': [], 'test_f1': [], 'nodes': [], 'depth': []}

# Run cross-validation
for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
    print(f"Fold {fold}/{n_folds}...")
    
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    # Standardize
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # ===== GA Training =====
    n_features = X_train.shape[1]
    n_classes = len(np.unique(y))
    feature_ranges = {i: (X_train[:, i].min(), X_train[:, i].max()) 
                     for i in range(n_features)}
    
    ga_config = GAConfig(
        population_size=80,
        n_generations=40,
        crossover_prob=0.72,
        mutation_prob=0.18,
        tournament_size=4,
        elitism_ratio=0.12
    )
    
    initializer = TreeInitializer(
        n_features=n_features,
        n_classes=n_classes,
        max_depth=6,
        min_samples_split=8,
        min_samples_leaf=3
    )
    
    fitness_calc = FitnessCalculator(
        mode='weighted_sum',
        accuracy_weight=0.68,
        interpretability_weight=0.32
    )
    
    mutation = Mutation(n_features=n_features, feature_ranges=feature_ranges)
    
    ga_engine = GAEngine(ga_config, initializer, 
                        fitness_calc.calculate_fitness, mutation)
    best_tree = ga_engine.evolve(X_train, y_train, verbose=False)
    
    # GA Evaluation
    predictor = TreePredictor()
    y_pred_ga = predictor.predict(best_tree, X_test)
    
    ga_results['test_acc'].append(accuracy_score(y_test, y_pred_ga))
    ga_results['test_f1'].append(f1_score(y_test, y_pred_ga, average='weighted'))
    ga_results['nodes'].append(best_tree.get_num_nodes())
    ga_results['depth'].append(best_tree.get_depth())
    
    # ===== CART Baseline =====
    cart = DecisionTreeClassifier(max_depth=6, random_state=42)
    cart.fit(X_train, y_train)
    y_pred_cart = cart.predict(X_test)
    
    cart_results['test_acc'].append(accuracy_score(y_test, y_pred_cart))
    cart_results['test_f1'].append(f1_score(y_test, y_pred_cart, average='weighted'))
    cart_results['nodes'].append(cart.tree_.node_count)
    cart_results['depth'].append(cart.tree_.max_depth)
    
    print(f"  GA: Acc={ga_results['test_acc'][-1]:.3f}, Nodes={ga_results['nodes'][-1]}")
    print(f"  CART: Acc={cart_results['test_acc'][-1]:.3f}, Nodes={cart_results['nodes'][-1]}")

# ===== Statistical Analysis =====
print("\n" + "="*60)
print("RESULTS SUMMARY")
print("="*60)

# Compute means and stds
ga_acc_mean = np.mean(ga_results['test_acc'])
ga_acc_std = np.std(ga_results['test_acc'])
cart_acc_mean = np.mean(cart_results['test_acc'])
cart_acc_std = np.std(cart_results['test_acc'])

print(f"\nGA Accuracy: {ga_acc_mean:.4f} ± {ga_acc_std:.4f}")
print(f"CART Accuracy: {cart_acc_mean:.4f} ± {cart_acc_std:.4f}")

ga_nodes_mean = np.mean(ga_results['nodes'])
cart_nodes_mean = np.mean(cart_results['nodes'])
reduction = (1 - ga_nodes_mean / cart_nodes_mean) * 100

print(f"\nGA Nodes: {ga_nodes_mean:.1f}")
print(f"CART Nodes: {cart_nodes_mean:.1f}")
print(f"Size Reduction: {reduction:.1f}%")

# Statistical test
t_stat, p_value = stats.ttest_rel(ga_results['test_acc'], cart_results['test_acc'])
pooled_std = np.sqrt((np.var(ga_results['test_acc']) + np.var(cart_results['test_acc'])) / 2)
cohens_d = (ga_acc_mean - cart_acc_mean) / pooled_std if pooled_std > 0 else 0.0

print(f"\nPaired t-test:")
print(f"  t-statistic: {t_stat:.4f}")
print(f"  p-value: {p_value:.4f}")
print(f"  Cohen's d: {cohens_d:.4f}")

if p_value > 0.05:
    print(f"  Result: No significant difference (statistically equivalent)")
else:
    winner = "GA" if ga_acc_mean > cart_acc_mean else "CART"
    print(f"  Result: {winner} is significantly better (p < 0.05)")
```

## Experiment Outputs

### 1. Results CSV

Location: `results/results_FAST_YYYYMMDD_HHMMSS.csv`

```csv
Dataset,Model,Test Acc,Test F1,Time (s),Nodes,Depth
iris,GA-Optimized,0.9455 ± 0.0807,0.9452 ± 0.0811,3.41,7.4,2.4
iris,CART,0.9241 ± 0.1043,0.9227 ± 0.1068,0.01,16.4,4.4
iris,Random Forest,0.9533 ± 0.0340,0.9532 ± 0.0341,0.46,,,
wine,GA-Optimized,0.8819 ± 0.1039,0.8789 ± 0.1063,5.50,10.7,3.0
wine,CART,0.8722 ± 0.1070,0.8687 ± 0.1099,0.01,20.7,4.4
...
```

### 2. Configuration YAML

Location: `results/config_FAST_YYYYMMDD_HHMMSS.yaml`

Saves the exact configuration used for reproducibility.

### 3. Console Output

```
======================================================================
Running FAST GA on iris
======================================================================
  Fold 1/20... Acc=0.933, Nodes=7, Time=3.2s
  Fold 2/20... Acc=0.967, Nodes=7, Time=3.1s
  ...
  Fold 20/20... Acc=0.933, Nodes=9, Time=3.4s

======================================================================
Running CART on iris
======================================================================
  Fold 1/20... Acc=0.900, Time=0.00s
  ...

======================================================================
FINAL RESULTS
======================================================================

Dataset              Model           Test Acc              Nodes  Depth
--------------------------------------------------------------------------------
iris                 GA-Optimized    0.9455 ± 0.0807      7.4    2.4  
iris                 CART            0.9241 ± 0.1043      16.4   4.4  
iris                 Random Forest   0.9533 ± 0.0340      N/A    N/A  

======================================================================
Tree Size Analysis (GA vs CART)
======================================================================

iris                : GA=  7.4, CART= 16.4, Ratio=0.45x  ✓✓ Much smaller
wine                : GA= 10.7, CART= 20.7, Ratio=0.52x  ✓ Smaller
breast_cancer       : GA=  6.5, CART= 35.5, Ratio=0.18x  ✓✓ Much smaller

======================================================================
Statistical Tests (GA vs CART)
======================================================================

iris                : t= 1.371, p=0.1864 ns, d= 0.230
wine                : t= 0.415, p=0.6831 ns, d= 0.092
breast_cancer       : t=-0.475, p=0.6402 ns, d=-0.108
```

## Statistical Analysis

### Paired t-Test

Tests whether GA and CART have significantly different accuracies:

```python
from scipy import stats

t_stat, p_value = stats.ttest_rel(ga_accuracies, cart_accuracies)

print(f"t-statistic: {t_stat:.4f}")
print(f"p-value: {p_value:.4f}")

if p_value > 0.05:
    print("No significant difference (p > 0.05)")
else:
    print("Significant difference (p < 0.05)")
```

### Effect Size (Cohen's d)

Measures magnitude of difference:

```python
pooled_std = np.sqrt((np.var(ga_acc) + np.var(cart_acc)) / 2)
cohens_d = (np.mean(ga_acc) - np.mean(cart_acc)) / pooled_std

print(f"Cohen's d: {cohens_d:.4f}")

if abs(cohens_d) < 0.2:
    print("Effect size: Negligible")
elif abs(cohens_d) < 0.5:
    print("Effect size: Small")
elif abs(cohens_d) < 0.8:
    print("Effect size: Medium")
else:
    print("Effect size: Large")
```

**Interpretation:**

- |d| < 0.2: Negligible difference
- 0.2 ≤ |d| < 0.5: Small effect
- 0.5 ≤ |d| < 0.8: Medium effect
- |d| ≥ 0.8: Large effect

### Confidence Intervals

```python
from scipy import stats

# 95% confidence interval
confidence = 0.95
n = len(ga_accuracies)
mean = np.mean(ga_accuracies)
std_err = stats.sem(ga_accuracies)
ci = std_err * stats.t.ppf((1 + confidence) / 2, n - 1)

print(f"Mean: {mean:.4f}")
print(f"95% CI: [{mean - ci:.4f}, {mean + ci:.4f}]")
```

## Advanced Experiment Configurations

### 1. Custom Dataset Experiment

```python
def run_custom_dataset_experiment(X, y, dataset_name, config, n_folds=10):
    """Run experiment on custom dataset."""
    
    print(f"\n{'='*70}")
    print(f"Running Experiment: {dataset_name}")
    print(f"{'='*70}")
    
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    results = {'test_acc': [], 'test_f1': [], 'nodes': [], 'depth': []}
    
    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
        print(f"  Fold {fold}/{n_folds}...", end=' ')
        
        # ... training logic ...
        
        print(f"Acc={acc:.3f}, Nodes={nodes}")
    
    # Statistical summary
    print(f"\nResults for {dataset_name}:")
    print(f"  Accuracy: {np.mean(results['test_acc']):.4f} ± {np.std(results['test_acc']):.4f}")
    print(f"  F1 Score: {np.mean(results['test_f1']):.4f} ± {np.std(results['test_f1']):.4f}")
    print(f"  Avg Nodes: {np.mean(results['nodes']):.1f}")
    print(f"  Avg Depth: {np.mean(results['depth']):.1f}")
    
    return results

# Example usage
from sklearn.datasets import load_digits
X, y = load_digits(return_X_y=True)
results = run_custom_dataset_experiment(X, y, "Digits", config)
```

### 2. Multiple Configuration Comparison

```python
configs = {
    'balanced': 'configs/balanced.yaml',
    'accuracy_focused': 'configs/custom.yaml',
    'interpretability_focused': 'configs/diverse.yaml'
}

all_results = {}

for config_name, config_path in configs.items():
    print(f"\n{'='*70}")
    print(f"Testing Configuration: {config_name}")
    print(f"{'='*70}")
    
    config = load_config(config_path)
    results = run_ga_experiment(X, y, "Breast Cancer", config, n_folds=5)
    all_results[config_name] = results

# Compare configurations
print("\n" + "="*70)
print("CONFIGURATION COMPARISON")
print("="*70)

for config_name, results in all_results.items():
    acc = np.mean(results['test_acc'])
    nodes = np.mean(results['nodes'])
    print(f"{config_name:20s}: Acc={acc:.4f}, Nodes={nodes:.1f}")
```

### 3. Hyperparameter Sensitivity Analysis

```python
# Test different population sizes
population_sizes = [30, 50, 80, 100, 150]
results_by_pop = {}

for pop_size in population_sizes:
    print(f"\nTesting population_size={pop_size}")
    
    # Modify config
    config['ga']['population_size'] = pop_size
    
    # Run experiment (3 folds for speed)
    results = run_ga_experiment(X, y, "Iris", config, n_folds=3)
    results_by_pop[pop_size] = np.mean(results['test_acc'])

# Plot results
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.plot(population_sizes, list(results_by_pop.values()), 
         marker='o', linewidth=2, markersize=8)
plt.xlabel('Population Size')
plt.ylabel('Mean Test Accuracy')
plt.title('Sensitivity to Population Size')
plt.grid(True, alpha=0.3)
plt.savefig('results/sensitivity_population.png')
```

## Reproducibility

### Ensuring Reproducible Results

```python
import random
import numpy as np

def set_seed(seed=42):
    """Set all random seeds."""
    random.seed(seed)
    np.random.seed(seed)
    # If using torch/tensorflow in future:
    # torch.manual_seed(seed)
    # tf.random.set_seed(seed)

# Always set seed before experiments
set_seed(42)

# Run experiments
results = run_ga_experiment(X, y, "Breast Cancer", config)
```

### Logging Experiment Details

```python
import yaml
import json
from datetime import datetime
from pathlib import Path

def save_experiment_metadata(config, results, output_dir='results'):
    """Save complete experiment metadata."""
    
    metadata = {
        'timestamp': datetime.now().isoformat(),
        'config': config,
        'results_summary': {
            'mean_accuracy': float(np.mean(results['test_acc'])),
            'std_accuracy': float(np.std(results['test_acc'])),
            'mean_nodes': float(np.mean(results['nodes'])),
            'mean_depth': float(np.mean(results['depth'])),
        },
        'system_info': {
            'python_version': sys.version,
            'numpy_version': np.__version__,
            'sklearn_version': sklearn.__version__,
        }
    }
    
    # Save as JSON
    output_path = Path(output_dir) / f'experiment_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(output_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✓ Experiment metadata saved: {output_path}")
```

## Performance Tips

### 1. Parallel Fold Execution

```python
from joblib import Parallel, delayed

def run_fold(fold_data, config):
    """Run single fold."""
    train_idx, test_idx = fold_data
    # ... training logic ...
    return results

# Parallel execution
fold_results = Parallel(n_jobs=-1)(
    delayed(run_fold)(fold_data, config) 
    for fold_data in skf.split(X, y)
)
```

### 2. Reduce Generations for Quick Tests

```bash
# Quick test (2-3 minutes total)
python scripts/experiment.py --config configs/custom.yaml --cv-folds 3
```

Edit config for fast testing:

```yaml
ga:
  population_size: 30  # Reduced from 80
  n_generations: 10    # Reduced from 40
```

## Next Steps

- **Visualize Results**: See Visualization Guide
- **Hyperparameter Tuning**: Use Optuna for optimization
- **Pareto Analysis**: Explore Multi-Objective Optimization
- **Statistical Methods**: Deep dive into Statistical Tests