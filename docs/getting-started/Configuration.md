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

The framework includes several standardized configuration files:

| Config File | Purpose | Best For | Key Features |
|-------------|---------|----------|--------------|
| **`paper.yaml`** ⭐ | Research paper settings | Replicating results | 24-77% size reduction |
| `fast.yaml` | Quick experiments | Development, testing | 3-5× faster |
| `balanced.yaml` | Equal objectives | General exploration | 50/50 balance |
| `accuracy_focused.yaml` | Max performance | Production systems | 85% accuracy weight |
| `interpretability_focused.yaml` | Max transparency | Medical, legal | 60% interp weight |
| `default.yaml` | Standard settings | First-time users | Good defaults |

### 📄 **Paper Configuration (Recommended)**

The `paper.yaml` configuration contains the **exact hyperparameters used in the research paper**:

```yaml
# configs/paper.yaml - Research paper settings
ga:
  population_size: 80
  n_generations: 40
  crossover_prob: 0.72
  mutation_prob: 0.18
  tournament_size: 4
  elitism_ratio: 0.12

fitness:
  weights:
    accuracy: 0.68
    interpretability: 0.32
  
  interpretability_weights:
    node_complexity: 0.50
    feature_coherence: 0.10
    tree_balance: 0.10
    semantic_coherence: 0.30

tree:
  max_depth: 6
  min_samples_split: 8
  min_samples_leaf: 3
```

**Results achieved:**
- Iris: 55% size reduction
- Wine: 48% size reduction  
- Breast Cancer: 82% size reduction
- All with statistically equivalent accuracy (p > 0.05)

## Quick Start Examples

```bash
# Use paper configuration (recommended)
python scripts/experiment.py --config configs/paper.yaml

# Quick test
python scripts/train.py --config configs/fast.yaml --dataset iris

# Balance accuracy and interpretability
python scripts/experiment.py --config configs/balanced.yaml

# Maximum interpretability
python scripts/train.py --config configs/interpretability_focused.yaml --dataset breast_cancer
```

## Configuration Details

### GA Parameters

```yaml
ga:
  population_size: 80          # Number of trees per generation
  n_generations: 40            # Number of evolution iterations
  crossover_prob: 0.72         # Probability of crossover (0-1)
  mutation_prob: 0.18          # Probability of mutation (0-1)
  tournament_size: 4           # Selection pressure (2-7)
  elitism_ratio: 0.12          # Top % preserved (0-0.3)
  
  mutation_types:              # Must sum to 1.0
    threshold_perturbation: 0.45
    feature_replacement: 0.25
    prune_subtree: 0.25
    expand_leaf: 0.05
```

### Tree Constraints

```yaml
tree:
  max_depth: 6                 # Maximum tree depth (3-10)
  min_samples_split: 8         # Min samples to split (2-20)
  min_samples_leaf: 3          # Min samples in leaf (1-10)
```

### Fitness Weights

```yaml
fitness:
  mode: weighted_sum           # 'weighted_sum' or 'pareto'
  weights:                     # Must sum to 1.0
    accuracy: 0.68
    interpretability: 0.32
  
  interpretability_weights:    # Must sum to 1.0
    node_complexity: 0.50      # Penalize tree size
    feature_coherence: 0.10    # Reward feature reuse
    tree_balance: 0.10         # Prefer balanced trees
    semantic_coherence: 0.30   # Consistent predictions
```

## Choosing the Right Configuration

### By Use Case

| Use Case | Configuration | Why |
|----------|--------------|-----|
| **Research replication** | `paper.yaml` | Exact paper parameters |
| **Quick testing** | `fast.yaml` | Faster iterations |
| **Production deployment** | `accuracy_focused.yaml` | Max performance |
| **Medical diagnosis** | `interpretability_focused.yaml` | Transparency required |
| **General use** | `balanced.yaml` | Good starting point |

### By Priority

**Accuracy is most important** (Competitions, production):
```bash
python scripts/train.py --config configs/accuracy_focused.yaml
```

**Interpretability is most important** (Healthcare, legal):
```bash
python scripts/train.py --config configs/interpretability_focused.yaml
```

**Equal priority** (Exploration):
```bash
python scripts/train.py --config configs/balanced.yaml
```

## Creating Custom Configurations

### Method 1: Copy and Modify

```bash
# Copy paper config as starting point
cp configs/paper.yaml configs/my_custom.yaml

# Edit my_custom.yaml with your preferences
nano configs/my_custom.yaml

# Use your config
python scripts/train.py --config configs/my_custom.yaml --dataset wine
```

### Method 2: Create from Scratch

```yaml
# configs/my_custom.yaml

# GA settings
ga:
  population_size: 100        # Your choice
  n_generations: 50
  crossover_prob: 0.75
  mutation_prob: 0.20
  tournament_size: 3
  elitism_ratio: 0.15
  
  mutation_types:
    threshold_perturbation: 0.45
    feature_replacement: 0.25
    prune_subtree: 0.25
    expand_leaf: 0.05

# Tree constraints
tree:
  max_depth: 7                # Deeper trees
  min_samples_split: 10
  min_samples_leaf: 4

# Fitness function
fitness:
  mode: weighted_sum
  weights:
    accuracy: 0.75            # More accuracy focus
    interpretability: 0.25
  
  interpretability_weights:
    node_complexity: 0.60
    feature_coherence: 0.20
    tree_balance: 0.10
    semantic_coherence: 0.10

# Experiment settings
experiment:
  datasets:
    - iris
    - wine
  cv_folds: 10
  random_state: 42
```

## Parameter Guidelines

### Population Size
- **Small (30-50)**: Fast, may miss optimal solutions
- **Medium (50-100)**: Good balance ✓ (paper uses 80)
- **Large (100-200)**: Better exploration, slower

### Generations
- **Few (20-30)**: Quick experiments
- **Medium (30-50)**: Standard ✓ (paper uses 40)
- **Many (50-100)**: Thorough optimization

### Accuracy Weight
- **0.85-0.95**: Maximum accuracy (production)
- **0.68-0.75**: Balanced ✓ (paper uses 0.68)
- **0.50-0.60**: High interpretability (medical)

### Max Depth
- **3-4**: Very interpretable
- **5-7**: Good balance ✓ (paper uses 6)
- **8-10**: More complex, less interpretable

## Configuration Presets Summary

```bash
# Paper configuration (recommended for research)
configs/paper.yaml
  - Exact research paper parameters
  - 24-77% size reduction achieved
  - Statistical equivalence proven

# Fast configuration (development)
configs/fast.yaml
  - 50 population, 30 generations
  - 3-5× faster than paper config
  - Good for quick iteration

# Balanced configuration (exploration)
configs/balanced.yaml
  - 50/50 accuracy-interpretability
  - 100 population, 40 generations
  - Good starting point

# Accuracy-focused (production)
configs/accuracy_focused.yaml
  - 85% accuracy weight
  - Deeper trees allowed (depth 8)
  - Maximum performance

# Interpretability-focused (high-stakes)
configs/interpretability_focused.yaml
  - 60% interpretability weight
  - Aggressive pruning
  - Very small trees

# Default configuration (beginners)
configs/default.yaml
  - Simple, reasonable defaults
  - 70/30 accuracy-interpretability
  - Easy to understand
```

## Command-Line Overrides

Override config parameters from command line:

```bash
# Use paper config but change generations
python scripts/train.py --config configs/paper.yaml \
    --generations 60 \
    --dataset breast_cancer

# Override multiple parameters
python scripts/train.py --config configs/paper.yaml \
    --population 100 \
    --generations 50 \
    --max-depth 7 \
    --accuracy-weight 0.75
```

## Configuration Best Practices

### 1. Start with Paper Config

```bash
# Always start here for research
python scripts/experiment.py --config configs/paper.yaml
```

### 2. Use Fast Config for Development

```bash
# Quick iteration during development
python scripts/train.py --config configs/fast.yaml --dataset iris
```

### 3. Version Control Your Configs

```bash
git add configs/my_experiment.yaml
git commit -m "Add config for medical diagnosis experiment"
```

### 4. Document Your Changes

```yaml
# configs/my_experiment.yaml
# Author: Your Name
# Date: 2025-01-15
# Purpose: Optimize for medical diagnosis
# Changes from paper.yaml:
#   - Increased interpretability_weight to 0.45
#   - Reduced max_depth to 5
#   - Increased prune_subtree mutation to 0.35

ga:
  # ... your settings
```

## Troubleshooting

**Problem:** Trees too large  
**Solution:** Use `interpretability_focused.yaml` or increase interpretability weight

**Problem:** Accuracy too low  
**Solution:** Use `accuracy_focused.yaml` or increase accuracy weight

**Problem:** Training too slow  
**Solution:** Use `fast.yaml` or reduce population/generations

**Problem:** Results not reproducible  
**Solution:** Ensure `random_state: 42` in experiment section

## Next Steps

- [Training Guide](../user-guides/training.md) - Train with your config
- [Experiments Guide](../user-guides/experiments.md) - Run benchmarks
- [Hyperparameter Tuning](../user-guides/hyperparameter-tuning.md) - Optimize configs
- [FAQ](../faq/faq.md) - Common questions

## Quick Reference

```bash
# Research (paper replication)
python scripts/experiment.py --config configs/paper.yaml

# Development (fast iteration)
python scripts/train.py --config configs/fast.yaml --dataset iris

# Production (max accuracy)
python scripts/train.py --config configs/accuracy_focused.yaml

# Medical/Legal (max interpretability)
python scripts/train.py --config configs/interpretability_focused.yaml

# General (balanced)
python scripts/experiment.py --config configs/balanced.yaml
```