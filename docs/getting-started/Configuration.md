# Configuration Guide

This guide explains how to configure the GA-Optimized Decision Trees framework using YAML configuration files.

## Overview

All hyperparameters are externalized to YAML files for:
- **Reproducibility:** Same config = same results
- **Experimentation:** Easy parameter comparison
- **Version control:** Track configuration changes
- **Sharing:** Share configurations with collaborators

## Configuration Structure

A complete configuration file has four main sections:

```yaml
ga:           # Genetic algorithm parameters
tree:         # Tree constraints
fitness:      # Fitness function weights
experiment:   # Experiment settings
```

## Available Configurations

The framework includes several pre-configured files:

| Config File | Purpose | Best For |
|-------------|---------|----------|
| `configs/custom.yaml` | **Recommended** - Optimized settings | Most use cases |
| `configs/default.yaml` | Baseline configuration | Learning/testing |
| `configs/balanced.yaml` | Equal emphasis on objectives | Exploration |
| `configs/optimized.yaml` | Optuna-tuned parameters | Max performance |

## Detailed Configuration Reference

### 1. Genetic Algorithm (`ga`)

Controls the evolution process.

```yaml
ga:
  population_size: 80          # Number of individuals per generation
  n_generations: 40            # Number of evolution cycles
  crossover_prob: 0.72         # Probability of crossover (0-1)
  mutation_prob: 0.18          # Probability of mutation (0-1)
  tournament_size: 4           # Selection pressure (2-7)
  elitism_ratio: 0.12          # Top % preserved (0-0.3)
  
  mutation_types:              # Mutation operator probabilities (must sum to 1.0)
    threshold_perturbation: 0.45  # Adjust split thresholds
    feature_replacement: 0.25     # Change split features
    prune_subtree: 0.25          # Remove subtrees
    expand_leaf: 0.05            # Expand leaves to nodes
```

#### Parameter Guidelines

**population_size:**
- **Small (30-50):** Fast, may converge prematurely
- **Medium (50-100):** Good balance ✓
- **Large (100-200):** Better exploration, slower

**n_generations:**
- **Few (20-30):** Quick experiments
- **Medium (30-50):** Standard ✓
- **Many (50-100):** Thorough optimization

**crossover_prob:**
- **Low (0.5-0.6):** More mutation emphasis
- **Medium (0.7-0.8):** Balanced ✓
- **High (0.8-0.9):** More recombination

**mutation_prob:**
- **Low (0.1-0.15):** Exploitation focus
- **Medium (0.15-0.25):** Balanced ✓
- **High (0.25-0.35):** Exploration focus

**tournament_size:**
- **Small (2-3):** Weak selection pressure
- **Medium (3-5):** Standard ✓
- **Large (5-7):** Strong selection pressure

**Mutation Type Distribution:**
```yaml
# Interpretability-focused (more pruning)
mutation_types:
  threshold_perturbation: 0.40
  feature_replacement: 0.20
  prune_subtree: 0.35
  expand_leaf: 0.05

# Accuracy-focused (less pruning)
mutation_types:
  threshold_perturbation: 0.50
  feature_replacement: 0.30
  prune_subtree: 0.15
  expand_leaf: 0.05
```

### 2. Tree Constraints (`tree`)

Controls tree structure limits.

```yaml
tree:
  max_depth: 6                 # Maximum tree depth (3-10)
  min_samples_split: 8         # Min samples to split node (2-20)
  min_samples_leaf: 3          # Min samples in leaf (1-10)
  max_features: null           # Feature sampling (null, "sqrt", "log2", int)
```

#### Parameter Guidelines

**max_depth:**
- **Shallow (3-4):** Very interpretable, may underfit
- **Medium (5-7):** Good balance ✓
- **Deep (8-10):** More expressive, less interpretable

**min_samples_split:**
- **Low (2-5):** More granular splits
- **Medium (5-15):** Balanced ✓
- **High (15-30):** Simpler trees

**min_samples_leaf:**
- **Low (1-3):** Detailed leaves ✓
- **Medium (3-7):** Balanced
- **High (7-15):** Coarse predictions

**max_features:**
```yaml
max_features: null    # Use all features (default)
max_features: "sqrt"  # Use √n_features (Random Forest style)
max_features: "log2"  # Use log₂(n_features)
max_features: 5       # Use exactly 5 features
```

#### Dataset-Specific Constraints

```yaml
tree:
  max_depth: 6
  min_samples_split: 8
  min_samples_leaf: 3
  
  # Override for specific datasets
  dataset_specific:
    iris:
      max_depth: 4           # Simpler dataset = shallower tree
      min_samples_split: 5
    wine:
      max_depth: 5
      min_samples_split: 6
    breast_cancer:
      max_depth: 6           # Complex dataset = deeper tree
      min_samples_split: 10
```

### 3. Fitness Function (`fitness`)

Controls objective trade-offs.

```yaml
fitness:
  mode: weighted_sum           # 'weighted_sum' or 'pareto'
  
  weights:                     # Primary objectives (must sum to 1.0)
    accuracy: 0.68            # Weight for accuracy (0-1)
    interpretability: 0.32    # Weight for interpretability (0-1)
  
  interpretability_weights:    # Sub-objectives (must sum to 1.0)
    node_complexity: 0.55      # Penalize large trees
    feature_coherence: 0.25    # Reward feature reuse
    tree_balance: 0.10         # Prefer balanced trees
    semantic_coherence: 0.10   # Encourage consistent predictions
```

#### Objective Weight Presets

**Accuracy-Focused (90/10):**
```yaml
weights:
  accuracy: 0.90
  interpretability: 0.10
# Result: High accuracy, larger trees
```

**Balanced (70/30):**
```yaml
weights:
  accuracy: 0.70
  interpretability: 0.30
# Result: Good balance ✓
```

**Interpretability-Focused (50/50):**
```yaml
weights:
  accuracy: 0.50
  interpretability: 0.50
# Result: Very small trees, lower accuracy
```

**Custom (68/32 - Recommended):**
```yaml
weights:
  accuracy: 0.68
  interpretability: 0.32
# Result: Slight accuracy preference, still compact trees ✓
```

#### Interpretability Component Presets

**Size-Focused:**
```yaml
interpretability_weights:
  node_complexity: 0.70      # Heavy emphasis on small trees
  feature_coherence: 0.15
  tree_balance: 0.10
  semantic_coherence: 0.05
```

**Feature-Focused:**
```yaml
interpretability_weights:
  node_complexity: 0.30
  feature_coherence: 0.50    # Emphasis on feature reuse
  tree_balance: 0.10
  semantic_coherence: 0.10
```

**Balanced (Recommended):**
```yaml
interpretability_weights:
  node_complexity: 0.55      # Primary focus on tree size
  feature_coherence: 0.25    # Secondary focus on features
  tree_balance: 0.10
  semantic_coherence: 0.10
```

### 4. Experiment Settings (`experiment`)

Controls experimental setup.

```yaml
experiment:
  datasets:                    # List of datasets to use
    - iris
    - wine
    - breast_cancer
  
  cv_folds: 20                 # Cross-validation folds (5-20)
  n_repeats: 1                 # Repeated CV (1-5)
  test_size: 0.2               # Test set proportion (0.1-0.3)
  random_state: 42             # Random seed for reproducibility
  
  baselines:                   # Comparison models
    - cart
    - pruned_cart
    - random_forest
    - xgboost
  
  metrics:                     # Metrics to compute
    - accuracy
    - f1_weighted
    - precision
    - recall
```

## Complete Example Configurations

### Example 1: Quick Experimentation

For fast iteration during development:

```yaml
# configs/quick.yaml
ga:
  population_size: 30
  n_generations: 20
  crossover_prob: 0.7
  mutation_prob: 0.2
  tournament_size: 3
  elitism_ratio: 0.1

tree:
  max_depth: 4
  min_samples_split: 10
  min_samples_leaf: 5

fitness:
  mode: weighted_sum
  weights:
    accuracy: 0.7
    interpretability: 0.3
  interpretability_weights:
    node_complexity: 0.6
    feature_coherence: 0.2
    tree_balance: 0.1
    semantic_coherence: 0.1

experiment:
  datasets: [iris]  # Single dataset
  cv_folds: 5       # Fast CV
  random_state: 42
```

**Usage:**
```bash
python scripts/experiment.py --config configs/quick.yaml
# Runtime: ~2 minutes
```

### Example 2: Research-Quality Evaluation

For publication-ready results:

```yaml
# configs/research.yaml
ga:
  population_size: 100
  n_generations: 60
  crossover_prob: 0.75
  mutation_prob: 0.15
  tournament_size: 5
  elitism_ratio: 0.15

tree:
  max_depth: 7
  min_samples_split: 5
  min_samples_leaf: 2

fitness:
  mode: weighted_sum
  weights:
    accuracy: 0.65
    interpretability: 0.35
  interpretability_weights:
    node_complexity: 0.60
    feature_coherence: 0.20
    tree_balance: 0.10
    semantic_coherence: 0.10

experiment:
  datasets: [iris, wine, breast_cancer]
  cv_folds: 20      # High statistical rigor
  n_repeats: 3      # Repeated CV
  random_state: 42
```

**Usage:**
```bash
python scripts/experiment.py --config configs/research.yaml
# Runtime: ~2 hours
```

### Example 3: Domain-Specific (Healthcare)

Optimized for medical applications:

```yaml
# configs/healthcare.yaml
ga:
  population_size: 80
  n_generations: 50
  mutation_types:
    prune_subtree: 0.40       # Aggressive pruning for simplicity

tree:
  max_depth: 4                # Very shallow for interpretability
  min_samples_split: 15       # Conservative splits
  min_samples_leaf: 10        # Robust leaves

fitness:
  weights:
    accuracy: 0.55            # Prioritize interpretability
    interpretability: 0.45
  interpretability_weights:
    node_complexity: 0.70     # Extreme simplicity
    feature_coherence: 0.20
    tree_balance: 0.05
    semantic_coherence: 0.05
```

## Using Configurations

### Command Line

```bash
# Use specific config
python scripts/train.py --config configs/custom.yaml --dataset iris

# Override parameters
python scripts/train.py --config configs/custom.yaml \
    --generations 50 \
    --population 100 \
    --max-depth 7

# Run experiments
python scripts/experiment.py --config configs/custom.yaml

# Pareto optimization
python scripts/run_pareto_optimization.py --config configs/custom.yaml
```

### Python API

```python
import yaml
from ga_trees.ga.engine import GAConfig, TreeInitializer, GAEngine

# Load config
with open('configs/custom.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Use config
ga_config = GAConfig(
    population_size=config['ga']['population_size'],
    n_generations=config['ga']['n_generations'],
    # ... other parameters
)

# Or create config programmatically
ga_config = GAConfig(
    population_size=80,
    n_generations=40,
    crossover_prob=0.72,
    mutation_prob=0.18
)
```

## Configuration Best Practices

### 1. Start with Defaults
Begin with `configs/custom.yaml` and adjust incrementally.

### 2. Version Control
```bash
git add configs/my_experiment.yaml
git commit -m "Add config for medical diagnosis experiment"
```

### 3. Naming Convention
```
configs/
├── custom.yaml              # Your main config
├── experiment_name.yaml     # Specific experiment
├── dataset_name.yaml        # Dataset-specific
└── author_date.yaml         # Dated experiments
```

### 4. Documentation
Add comments to your configs:

```yaml
# Breast cancer classification - high interpretability focus
# Author: Your Name
# Date: 2025-11-28
# Goal: Maximize interpretability for clinical use

ga:
  population_size: 80  # Balanced exploration
  # ... rest of config
```

### 5. Parameter Tuning Order

1. **Start simple:** Use default config
2. **Adjust objectives:** Change accuracy/interpretability weights
3. **Tune tree constraints:** Adjust max_depth, min_samples
4. **Optimize GA:** Tune population_size, n_generations
5. **Fine-tune operators:** Adjust mutation types

## Troubleshooting

**Problem:** Trees too large  
**Solution:** Increase `interpretability` weight or `node_complexity` weight

**Problem:** Accuracy too low  
**Solution:** Increase `accuracy` weight or `max_depth`

**Problem:** Slow convergence  
**Solution:** Increase `population_size` or `n_generations`

**Problem:** Premature convergence  
**Solution:** Increase `mutation_prob` or decrease `elitism_ratio`

## Next Steps

- [Training Models](../user-guides/training.md)
- [Hyperparameter Tuning](../user-guides/hyperparameter-tuning.md)
- [API Reference](../api-reference/)
