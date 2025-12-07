# Hyperparameter Tuning with Optuna

Complete guide to hyperparameter optimization using Optuna for GA-optimized decision trees.

## Overview

Hyperparameter tuning finds optimal GA and fitness parameters to maximize performance. The framework provides:

- **Bayesian optimization** with TPE (Tree-structured Parzen Estimator)
- **Early stopping** via MedianPruner
- **Optimization presets** for common scenarios
- **Progress tracking** with MLflow/JSON logging
- **Automated config export** in YAML format

## Quick Start

```bash
# Quick optimization (10 trials, ~10 minutes)
python scripts/hyperopt_with_optuna.py --preset fast --dataset iris

# Balanced optimization (30 trials, ~30 minutes)
python scripts/hyperopt_with_optuna.py --preset balanced --dataset breast_cancer

# Thorough optimization (100 trials, ~2 hours)
python scripts/hyperopt_with_optuna.py --preset thorough --dataset wine
```

## Optimization Presets

### Available Presets

|Preset|Trials|Timeout|CV Folds|Use Case|
|---|---|---|---|---|
|`fast`|10|10 min|3|Quick testing|
|`balanced`|30|30 min|5|Standard optimization|
|`thorough`|100|2 hours|5|Comprehensive search|
|`interpretability_focused`|50|1 hour|5|Small tree priority|
|`accuracy_focused`|50|1 hour|5|Max accuracy priority|

### Preset Definitions

**Fast Preset** (`--preset fast`):

```python
{
    'n_trials': 10,
    'timeout': 600,  # 10 minutes
    'cv_folds': 3,
    'search_space': {
        'population_size': {'type': 'categorical', 'choices': [50, 80, 100]},
        'n_generations': {'type': 'categorical', 'choices': [20, 30, 40]},
        'crossover_prob': {'type': 'float', 'low': 0.6, 'high': 0.8},
        'mutation_prob': {'type': 'float', 'low': 0.1, 'high': 0.25},
        'accuracy_weight': {'type': 'float', 'low': 0.6, 'high': 0.75}
    }
}
```

**Balanced Preset** (`--preset balanced`):

```python
{
    'n_trials': 30,
    'timeout': 1800,  # 30 minutes
    'cv_folds': 5,
    'search_space': {
        'population_size': {'type': 'int', 'low': 50, 'high': 150, 'step': 10},
        'n_generations': {'type': 'int', 'low': 20, 'high': 60, 'step': 10},
        'crossover_prob': {'type': 'float', 'low': 0.5, 'high': 0.9},
        'mutation_prob': {'type': 'float', 'low': 0.1, 'high': 0.3},
        'tournament_size': {'type': 'int', 'low': 2, 'high': 5},
        'max_depth': {'type': 'int', 'low': 3, 'high': 7},
        'accuracy_weight': {'type': 'float', 'low': 0.6, 'high': 0.9},
        'node_complexity_weight': {'type': 'float', 'low': 0.4, 'high': 0.7}
    }
}
```

## Basic Usage

### 1. Simple Optimization

```bash
# Optimize on Breast Cancer dataset
python scripts/hyperopt_with_optuna.py --dataset breast_cancer --n-trials 50

# With specific preset
python scripts/hyperopt_with_optuna.py --preset balanced --dataset wine

# Custom timeout and CV folds
python scripts/hyperopt_with_optuna.py --dataset iris --n-trials 40 --timeout 1200 --cv-folds 10
```

### 2. Resume Previous Study

```bash
# First run
python scripts/hyperopt_with_optuna.py --study-name my_optimization --dataset breast_cancer --n-trials 50

# Resume later (adds 30 more trials)
python scripts/hyperopt_with_optuna.py --study-name my_optimization --resume --n-trials 30
```

### 3. Persistent Storage

```bash
# Save to SQLite database for persistence
python scripts/hyperopt_with_optuna.py \
    --dataset breast_cancer \
    --storage sqlite:///optuna_studies.db \
    --study-name breast_cancer_opt \
    --n-trials 100
```

## Optimization Process

### How It Works

1. **Define Search Space**: Specify ranges for hyperparameters
2. **Bayesian Sampling**: TPE suggests promising configurations
3. **Cross-Validation**: Evaluate on CV folds
4. **Early Stopping**: Prune unpromising trials
5. **Select Best**: Return configuration with highest score
6. **Export Config**: Save to YAML for reproducibility

### Objective Function

The optimization maximizes a composite score:

```python
def objective(trial):
    # Suggest hyperparameters
    population_size = trial.suggest_int('population_size', 50, 150, step=10)
    n_generations = trial.suggest_int('n_generations', 20, 60, step=10)
    accuracy_weight = trial.suggest_float('accuracy_weight', 0.6, 0.9)
    # ... more parameters ...
    
    # Train with CV
    scores = []
    for train_idx, val_idx in cv.split(X, y):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Train GA tree
        tree = train_ga_tree(X_train, y_train, params)
        
        # Evaluate
        score = accuracy_score(y_val, tree.predict(X_val))
        scores.append(score)
        
        # Report intermediate value for pruning
        trial.report(np.mean(scores), fold)
        if trial.should_prune():
            raise optuna.TrialPruned()
    
    # Composite score: 80% accuracy + 20% size penalty
    mean_accuracy = np.mean(scores)
    mean_tree_size = np.mean(tree_sizes)
    size_penalty = 1.0 - min(mean_tree_size / 50.0, 1.0)
    
    return 0.8 * mean_accuracy + 0.2 * size_penalty
```

## Custom Optimization

### 1. Define Custom Search Space

```python
import optuna

# Custom search space
def custom_objective(trial):
    # GA parameters
    params = {
        'population_size': trial.suggest_int('population_size', 30, 200, step=10),
        'n_generations': trial.suggest_int('n_generations', 10, 100, step=5),
        'crossover_prob': trial.suggest_float('crossover_prob', 0.5, 0.95),
        'mutation_prob': trial.suggest_float('mutation_prob', 0.05, 0.35),
        'tournament_size': trial.suggest_int('tournament_size', 2, 6),
        'elitism_ratio': trial.suggest_float('elitism_ratio', 0.05, 0.2),
        
        # Tree constraints
        'max_depth': trial.suggest_int('max_depth', 3, 8),
        'min_samples_split': trial.suggest_int('min_samples_split', 5, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 2, 10),
        
        # Fitness weights
        'accuracy_weight': trial.suggest_float('accuracy_weight', 0.5, 0.95),
        'node_complexity_weight': trial.suggest_float('node_complexity_weight', 0.3, 0.8),
        'feature_coherence_weight': trial.suggest_float('feature_coherence_weight', 0.05, 0.35),
    }
    
    # Train and evaluate
    score = evaluate_params(params, X, y, cv_folds=5)
    
    return score

# Run optimization
study = optuna.create_study(direction='maximize')
study.optimize(custom_objective, n_trials=50)

print(f"Best trial: {study.best_trial.number}")
print(f"Best value: {study.best_value:.4f}")
print(f"Best params: {study.best_params}")
```

### 2. Multi-Objective Optimization

Optimize accuracy AND interpretability simultaneously:

```python
def multi_objective(trial):
    # ... define params ...
    
    # Train tree
    tree = train_ga_tree(X_train, y_train, params)
    
    # Evaluate both objectives
    accuracy = accuracy_score(y_test, tree.predict(X_test))
    interpretability = 1.0 / (1.0 + tree.get_num_nodes() / 20.0)
    
    return accuracy, interpretability  # Return tuple

# Multi-objective study
study = optuna.create_study(directions=['maximize', 'maximize'])
study.optimize(multi_objective, n_trials=100)

# Get Pareto front
pareto_trials = [t for t in study.best_trials]
print(f"Found {len(pareto_trials)} Pareto-optimal solutions")
```

### 3. Conditional Search Spaces

Different parameters based on conditions:

```python
def conditional_objective(trial):
    # Choose fitness mode
    fitness_mode = trial.suggest_categorical('fitness_mode', ['weighted_sum', 'pareto'])
    
    if fitness_mode == 'weighted_sum':
        # Weighted sum specific parameters
        accuracy_weight = trial.suggest_float('accuracy_weight', 0.5, 0.95)
        interp_weight = 1.0 - accuracy_weight
    else:
        # Pareto mode doesn't need weights
        accuracy_weight = None
        interp_weight = None
    
    # Choose mutation strategy
    mutation_focus = trial.suggest_categorical('mutation_focus', ['balanced', 'pruning', 'exploration'])
    
    if mutation_focus == 'pruning':
        mutation_types = {
            'threshold_perturbation': 0.30,
            'feature_replacement': 0.20,
            'prune_subtree': 0.45,  # High pruning
            'expand_leaf': 0.05
        }
    elif mutation_focus == 'exploration':
        mutation_types = {
            'threshold_perturbation': 0.30,
            'feature_replacement': 0.40,  # High exploration
            'prune_subtree': 0.20,
            'expand_leaf': 0.10
        }
    else:  # balanced
        mutation_types = {
            'threshold_perturbation': 0.45,
            'feature_replacement': 0.25,
            'prune_subtree': 0.25,
            'expand_leaf': 0.05
        }
    
    # Evaluate with conditional params
    # ...
```

## Analyzing Results

### 1. View Optimization Progress

```python
import optuna

# Load study
study = optuna.load_study(
    study_name='my_optimization',
    storage='sqlite:///optuna_studies.db'
)

# Print summary
print(f"Number of trials: {len(study.trials)}")
print(f"Best trial: {study.best_trial.number}")
print(f"Best value: {study.best_value:.4f}")
print("\nBest hyperparameters:")
for key, value in study.best_params.items():
    print(f"  {key:25s}: {value}")

# Get statistics
if study.best_trial.user_attrs:
    print("\nPerformance metrics:")
    print(f"  Mean Accuracy: {study.best_trial.user_attrs['mean_accuracy']:.4f}")
    print(f"  Std Accuracy:  {study.best_trial.user_attrs['std_accuracy']:.4f}")
    print(f"  Mean Tree Size: {study.best_trial.user_attrs['mean_tree_size']:.1f}")
```

### 2. Visualize Optimization History

```python
from optuna.visualization import (
    plot_optimization_history,
    plot_param_importances,
    plot_parallel_coordinate,
    plot_slice
)

# Optimization history
fig = plot_optimization_history(study)
fig.write_html('results/figures/optuna_history.html')

# Parameter importance
fig = plot_param_importances(study)
fig.write_html('results/figures/optuna_importance.html')

# Parallel coordinate plot
fig = plot_parallel_coordinate(study)
fig.write_html('results/figures/optuna_parallel.html')

# Slice plot (parameter effects)
fig = plot_slice(study)
fig.write_html('results/figures/optuna_slice.html')
```

### 3. Parameter Importance

```python
import matplotlib.pyplot as plt

# Get parameter importances
importances = optuna.importance.get_param_importances(study)

# Sort by importance
params = list(importances.keys())
values = list(importances.values())
sorted_pairs = sorted(zip(values, params), reverse=True)
values, params = zip(*sorted_pairs)

# Plot
fig, ax = plt.subplots(figsize=(10, 6))
ax.barh(range(len(params)), values, color='skyblue', edgecolor='black')
ax.set_yticks(range(len(params)))
ax.set_yticklabels(params)
ax.set_xlabel('Importance', fontsize=12, fontweight='bold')
ax.set_title('Hyperparameter Importance', fontsize=14, fontweight='bold')
ax.grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig('results/figures/param_importance.png', dpi=300)
```

## Best Practices

### 1. Start with Fast Preset

```bash
# Quick exploration (5-10 minutes)
python scripts/hyperopt_with_optuna.py --preset fast --dataset iris

# Analyze results
# If promising, scale up to balanced
python scripts/hyperopt_with_optuna.py --preset balanced --dataset iris
```

### 2. Use Proper Search Spaces

**Good search spaces:**

- Reasonable ranges based on domain knowledge
- Logarithmic for learning rates: `suggest_float('lr', 1e-5, 1e-2, log=True)`
- Discrete steps for integers: `suggest_int('pop', 50, 150, step=10)`

**Bad search spaces:**

- Too wide: `suggest_int('population', 10, 1000)` → slow convergence
- Too narrow: `suggest_float('mutation', 0.18, 0.22)` → no diversity

### 3. Monitor Progress

```python
# Custom callback for progress tracking
class ProgressCallback:
    def __init__(self):
        self.best_value = -np.inf
    
    def __call__(self, study, trial):
        if trial.value > self.best_value:
            self.best_value = trial.value
            print(f"Trial {trial.number}: New best value = {trial.value:.4f}")
            print(f"  Params: {trial.params}")

# Use callback
study.optimize(objective, n_trials=50, callbacks=[ProgressCallback()])
```

### 4. Handle Failed Trials

```python
def robust_objective(trial):
    try:
        # ... training code ...
        return score
    except Exception as e:
        print(f"Trial {trial.number} failed: {e}")
        return 0.0  # Return low score for failed trials

# Or use catch parameter
study.optimize(objective, n_trials=50, catch=(Exception,))
```

### 5. Warm Start from Good Config

```python
# Load good starting point
import yaml
with open('configs/custom.yaml', 'r') as f:
    good_config = yaml.safe_load(f)

# Use as first trial
def objective_with_warmstart(trial):
    if trial.number == 0:
        # Use known good config as first trial
        params = {
            'population_size': good_config['ga']['population_size'],
            'n_generations': good_config['ga']['n_generations'],
            # ... other params ...
        }
    else:
        # Sample normally
        params = {
            'population_size': trial.suggest_int('population_size', 50, 150, step=10),
            # ...
        }
    
    return evaluate_params(params)
```

## Exporting Results

### 1. Save Best Configuration

After optimization, the best config is automatically saved:

```bash
python scripts/hyperopt_with_optuna.py --dataset breast_cancer --output configs/optimized.yaml
```

**Output** (`configs/optimized.yaml`):

```yaml
ga:
  population_size: 140
  n_generations: 50
  crossover_prob: 0.6733843409158413
  mutation_prob: 0.15297790351647228
  tournament_size: 2
  elitism_ratio: 0.12

tree:
  max_depth: 6
  min_samples_split: 18
  min_samples_leaf: 5

fitness:
  mode: weighted_sum
  weights:
    accuracy: 0.8619364624278322
    interpretability: 0.13806353757216783
  interpretability_weights:
    node_complexity: 0.50
    feature_coherence: 0.10
    tree_balance: 0.10
    semantic_coherence: 0.30

optimization:
  study_name: ga_opt_breast_cancer_20250101_120000
  n_trials: 100
  best_value: 0.9245
  best_trial_number: 67
  optimization_date: "2025-01-01T12:34:56"
```

### 2. Export Trial History

```python
import pandas as pd

# Convert trials to DataFrame
trials_df = study.trials_dataframe()

# Save to CSV
trials_df.to_csv('results/optimization/optuna_trials.csv', index=False)

# Key columns:
# - number: Trial number
# - value: Objective value
# - params_*: Hyperparameter values
# - state: COMPLETE, PRUNED, FAIL
```

### 3. Test Optimized Config

```bash
# Test the optimized configuration
python scripts/test_optimized_config.py
```

This will:

1. Load optimized config
2. Train with 5-fold CV
3. Compare against CART baseline
4. Show statistical tests
5. Save comparison results

## Troubleshooting

### Issue: Optimization Too Slow

**Solutions:**

1. Reduce `n_trials` (100 → 30)
2. Reduce `cv_folds` (5 → 3)
3. Use smaller search space
4. Enable pruning aggressively

```python
# Aggressive pruning
pruner = optuna.pruners.MedianPruner(
    n_startup_trials=2,  # Prune after 2 folds
    n_warmup_steps=1,    # Start pruning quickly
    interval_steps=1
)
```

### Issue: Poor Convergence

**Solutions:**

1. Increase `n_trials` (30 → 100)
2. Widen search space
3. Check for bugs in objective function
4. Use different sampler

```python
# Try RandomSampler for comparison
sampler = optuna.samplers.RandomSampler(seed=42)
study = optuna.create_study(sampler=sampler, direction='maximize')
```

### Issue: All Trials Pruned

**Solutions:**

1. Disable pruning temporarily
2. Increase `n_startup_trials`
3. Check objective function returns valid values

```python
# Disable pruning for debugging
study = optuna.create_study(
    pruner=optuna.pruners.NopPruner(),  # No pruning
    direction='maximize'
)
```

### Issue: Memory Errors

**Solutions:**

1. Reduce population size
2. Reduce CV folds
3. Clear population after each trial

```python
def memory_efficient_objective(trial):
    # ... training code ...
    
    # Clear memory
    import gc
    gc.collect()
    
    return score
```

## Advanced Topics

### 1. Distributed Optimization

Run trials in parallel across machines:

```bash
# Machine 1
python scripts/hyperopt_with_optuna.py \
    --storage mysql://user:pass@server/db \
    --study-name distributed_study \
    --n-trials 50

# Machine 2 (same study)
python scripts/hyperopt_with_optuna.py \
    --storage mysql://user:pass@server/db \
    --study-name distributed_study \
    --n-trials 50
```

### 2. Dynamic Search Space

Adjust search space during optimization:

```python
class DynamicCallback:
    def __init__(self, study):
        self.study = study
    
    def __call__(self, study, trial):
        # After 20 trials, narrow search space
        if trial.number == 20:
            # Analyze best trials
            best_trials = sorted(study.trials, key=lambda t: t.value, reverse=True)[:5]
            
            # Compute mean of best params
            best_pop = np.mean([t.params['population_size'] for t in best_trials])
            
            print(f"Narrowing search around population_size={best_pop:.0f}")
            
            # Next trials will focus around best_pop
            # (requires custom sampling logic)
```

### 3. Constraint Handling

Enforce constraints on hyperparameters:

```python
def constrained_objective(trial):
    accuracy_weight = trial.suggest_float('accuracy_weight', 0.5, 0.95)
    interp_weight = 1.0 - accuracy_weight  # Constraint: must sum to 1.0
    
    node_complexity = trial.suggest_float('node_complexity_weight', 0.3, 0.8)
    feature_coherence = trial.suggest_float('feature_coherence_weight', 0.05, 0.35)
    
    # Constraint: sub-weights must sum to 1.0
    remaining = 1.0 - node_complexity - feature_coherence
    if remaining < 0.1:
        # Infeasible, return low score
        return 0.0
    
    tree_balance = remaining * 0.5
    semantic_coherence = remaining * 0.5
    
    # ... train and evaluate ...
```

## Next Steps

- **Test Optimized Config**: See Testing Guide
- **Compare Configurations**: Run Experiments
- **Analyze Results**: Use Visualization
- **Production Deployment**: See Model Export