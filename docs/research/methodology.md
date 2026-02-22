# Research Methodology

## Experimental Design

### Datasets
- **Iris**: 150 samples, 4 features, 3 classes (simple)
- **Wine**: 178 samples, 13 features, 3 classes (medium)
- **Breast Cancer**: 569 samples, 30 features, 2 classes (complex)

### Cross-Validation
- **20-fold stratified CV** for statistical rigor
- Same folds for GA and baselines (paired testing)
- Random seed: 42 (reproducibility)

### Hyperparameters

**GA (configs/paper.yaml)**:
```yaml
population_size: 80
n_generations: 40
crossover_prob: 0.72
mutation_prob: 0.18
accuracy_weight: 0.68
interpretability_weight: 0.32
max_depth: 6
```

**CART Baseline**:
```python
DecisionTreeClassifier(max_depth=6, random_state=42)
```

### Statistical Tests
- **Paired t-test**: Compare GA vs CART on same folds
- **Cohen's d**: Measure effect size
- **Significance level**: α = 0.05