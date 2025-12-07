# Performance Tips

Optimize training speed and resource usage.

## Speed Optimization

### 1. Reduce Population Size

```yaml
# Faster (30-50 individuals)
ga:
  population_size: 40
  n_generations: 30

# Slower but better results (80-120 individuals)
ga:
  population_size: 100
  n_generations: 50
```

### 2. Reduce Generations

Monitor convergence and stop early if fitness plateaus.

### 3. Use Smaller Trees

```yaml
tree:
  max_depth: 4        # Instead of 6
  min_samples_split: 15  # Instead of 8
```

### 4. Fewer CV Folds

```yaml
experiment:
  cv_folds: 5  # Instead of 20 for quick experiments
```

## Memory Optimization

### Large Datasets

```python
# Sample data for faster iteration
from sklearn.model_selection import train_test_split

# Use 20% of data for development
X_sample, _, y_sample, _ = train_test_split(
    X, y, train_size=0.2, stratify=y
)
```

### Large Populations

Monitor memory usage and reduce if needed:
```bash
# Monitor during training
watch -n 1 free -h
```

## Parallel Fitness Evaluation

Future feature - currently sequential.
