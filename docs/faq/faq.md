# Frequently Asked Questions

## General Questions

### What is this framework for?

This framework evolves decision trees using genetic algorithms to balance **accuracy** and **interpretability**. Unlike traditional algorithms like CART that only optimize for accuracy, our approach allows explicit control over tree complexity and explainability.

**Use cases:**
- Healthcare: Interpretable diagnostic models
- Finance: Transparent credit scoring
- Legal: Explainable classification systems
- Any domain requiring human-understandable models

### How is this different from CART or Random Forest?

| Aspect | CART | Random Forest | GA-Optimized |
|--------|------|---------------|--------------|
| Optimization | Greedy (local) | Ensemble | Global (evolutionary) |
| Objectives | Accuracy only | Accuracy only | Multi-objective |
| Interpretability | No control | Black box | Explicit control ✓ |
| Tree size | Often large | N/A (ensemble) | Controllable ✓ |
| Training time | Very fast | Fast | Moderate |

### What are the main achievements?

- **46-82% smaller trees** than CART on benchmark datasets
- **Statistically equivalent accuracy** (all p-values > 0.05)
- **Explicit interpretability control** via multi-objective optimization
- **Configuration-driven** experiments for reproducibility

### Is this ready for production use?

The framework is research-oriented and ideal for:
- ✓ Research projects
- ✓ Proof-of-concept applications
- ✓ Interpretable model development
- ✓ Comparative studies

For production deployment:
- Consider training time (minutes vs seconds for CART)
- Implement model versioning and monitoring
- Add input validation and error handling
- Consider using trained models offline

## Installation & Setup

### What Python version is required?

Python 3.8 or higher. Tested on 3.8, 3.9, 3.10, and 3.11.

```bash
python --version  # Check your version
```

### Installation fails with "ModuleNotFoundError"

Make sure you install in editable mode:

```bash
pip install -e .
```

This creates the necessary package structure. If still failing:

```bash
# Clean install
pip uninstall ga-optimized-trees
pip cache purge
pip install -r requirements.txt
pip install -e .
```

### Do I need GPU support?

No. The framework runs efficiently on CPU. GPU acceleration is not implemented as the bottleneck is typically the genetic algorithm logic, not matrix operations.

### Can I use this on Windows?

Yes! The framework works on Windows, macOS, and Linux. On Windows:

```bash
# Use PowerShell or Command Prompt
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

## Training & Usage

### How long does training take?

Depends on configuration and dataset:

**Quick experiments:**
```yaml
# configs/quick.yaml
population_size: 30
n_generations: 20
# Runtime: 1-3 minutes per dataset
```

**Standard experiments:**
```yaml
# configs/paper.yaml
population_size: 80
n_generations: 40
# Runtime: 5-10 minutes per dataset
```

**Research-quality:**
```yaml
# configs/research.yaml
population_size: 100
n_generations: 60
cv_folds: 20
# Runtime: 30-60 minutes per dataset
```

### How do I speed up training?

1. **Reduce population size:**
```yaml
population_size: 50  # Instead of 100
```

2. **Reduce generations:**
```yaml
n_generations: 30  # Instead of 50
```

3. **Use fewer CV folds:**
```yaml
cv_folds: 5  # Instead of 20
```

4. **Reduce tree depth:**
```yaml
max_depth: 4  # Instead of 6
```

### My trees are too large. How do I make them smaller?

Increase interpretability emphasis:

```yaml
fitness:
  weights:
    accuracy: 0.50        # Reduce from 0.68
    interpretability: 0.50  # Increase from 0.32
  
  interpretability_weights:
    node_complexity: 0.70  # Increase from 0.55
```

Or increase pruning mutation:

```yaml
mutation_types:
  prune_subtree: 0.40  # Increase from 0.25
```

### My accuracy is too low. How do I improve it?

1. **Increase accuracy weight:**
```yaml
fitness:
  weights:
    accuracy: 0.80
    interpretability: 0.20
```

2. **Allow deeper trees:**
```yaml
tree:
  max_depth: 8  # Instead of 6
```

3. **Increase population/generations:**
```yaml
ga:
  population_size: 120
  n_generations: 60
```

4. **Try hyperparameter tuning:**
```bash
python scripts/hyperopt_with_optuna.py --n-trials 30
```

### Can I use my own dataset?

Yes! Two approaches:

**Option 1: Python API**
```python
import numpy as np
from ga_trees.ga.engine import GAEngine, GAConfig, TreeInitializer, Mutation
from ga_trees.fitness.calculator import FitnessCalculator

# Load your data
X_train, y_train = load_your_data()

# Setup
n_features = X_train.shape[1]
n_classes = len(np.unique(y_train))
feature_ranges = {i: (X_train[:, i].min(), X_train[:, i].max()) 
                 for i in range(n_features)}

config = GAConfig(population_size=80, n_generations=40)
initializer = TreeInitializer(n_features=n_features, n_classes=n_classes,
                             max_depth=6, min_samples_split=8, min_samples_leaf=3)
fitness_calc = FitnessCalculator(accuracy_weight=0.68, interpretability_weight=0.32)
mutation = Mutation(n_features=n_features, feature_ranges=feature_ranges)

# Train
ga_engine = GAEngine(config, initializer, fitness_calc.calculate_fitness, mutation)
best_tree = ga_engine.evolve(X_train, y_train, verbose=True)
```

**Option 2: Add to data loader**

See [Custom Dataset Guide](../data/dataset-loader.md)

### How do I save and load models?

Models are automatically saved during training:

```bash
python scripts/train.py --config configs/paper.yaml --dataset iris
# Saves to: models/best_tree.pkl
```

Load model:

```python
import pickle

with open('models/best_tree.pkl', 'rb') as f:
    model_data = pickle.load(f)

tree = model_data['tree']
scaler = model_data['scaler']  # If standardization was used

# Make predictions
from ga_trees.fitness.calculator import TreePredictor
predictor = TreePredictor()

X_test = scaler.transform(X_test) if scaler else X_test
y_pred = predictor.predict(tree, X_test)
```

## Configuration

### Which config file should I use?

- **`configs/paper.yaml`** - Recommended for most use cases ✓
- **`configs/default.yaml`** - Learning/testing
- **`configs/balanced.yaml`** - Equal objectives
- **`configs/optimized.yaml`** - Optuna-tuned

### How do I override config parameters?

Command line overrides config file:

```bash
python scripts/train.py --config configs/paper.yaml \
    --generations 50 \
    --population 100 \
    --max-depth 7
```

### What's the best fitness weight ratio?

Depends on your priorities:

| Use Case | Accuracy Weight | Interpretability Weight |
|----------|-----------------|-------------------------|
| Max accuracy | 0.90 | 0.10 |
| Balanced (recommended) | 0.68 | 0.32 |
| Max interpretability | 0.50 | 0.50 |
| Healthcare/Legal | 0.55 | 0.45 |

### How many generations is enough?

Monitor convergence:

```python
history = ga_engine.get_history()
plt.plot(history['best_fitness'])
plt.xlabel('Generation')
plt.ylabel('Fitness')
plt.show()
```

If fitness plateaus before max generations, you can reduce:

```yaml
n_generations: 30  # If plateaus by generation 25
```

## Results & Evaluation

### How do I interpret the results?

```
Dataset          Model            Test Acc        Nodes
iris             GA-Optimized     94.55 ± 8.07%   7.4
iris             CART             92.41 ± 10.43%  16.4
```

**Interpretation:**
- GA accuracy: 94.55% (± 8.07% std)
- GA produces 7.4 nodes on average
- CART produces 16.4 nodes (55% larger)
- GA has +2.1% higher accuracy

### What does "p > 0.05" mean?

p-value > 0.05 means **no statistically significant difference**:

```
p = 0.640: No significant difference ✓
p = 0.03:  Significant difference ✗
```

Our results show p > 0.05 for all datasets, meaning GA performs **equivalently** to CART statistically.

### Why is my GA accuracy lower than CART?

This is expected when prioritizing interpretability:

```yaml
# If GA accuracy is 2-3% lower
fitness:
  weights:
    accuracy: 0.68      # 68% emphasis
    interpretability: 0.32  # 32% on simplicity
```

The trade-off:
- **2-3% accuracy loss** for **50-80% smaller trees**

If unacceptable, increase accuracy weight:

```yaml
weights:
  accuracy: 0.80  # More emphasis on accuracy
  interpretability: 0.20
```

### How do I visualize the Pareto front?

```bash
python scripts/run_pareto_optimization.py --config configs/paper.yaml
```

Creates `results/figures/pareto_front.png` showing accuracy vs interpretability trade-offs.

### Can I compare with XGBoost/LightGBM?

Currently, the framework compares with:
- CART (single tree)
- Random Forest (ensemble)
- XGBoost (ensemble)

To add custom baselines, see `src/ga_trees/baselines/baseline_models.py`.

## Advanced Topics

### Can I implement custom fitness functions?

Yes! Extend `FitnessCalculator`:

```python
from ga_trees.fitness.calculator import FitnessCalculator

class MyCustomFitness(FitnessCalculator):
    def calculate_fitness(self, tree, X, y):
        # Your custom logic
        accuracy = super().calculate_fitness(tree, X, y)
        custom_metric = self.my_custom_metric(tree)
        
        return 0.7 * accuracy + 0.3 * custom_metric
    
    def my_custom_metric(self, tree):
        # Example: reward trees with specific features
        preferred_features = {0, 2, 4}
        features_used = tree.get_features_used()
        return len(features_used & preferred_features) / len(preferred_features)
```

### Can I add custom mutation operators?

Yes! Extend `Mutation`:

```python
from ga_trees.ga.engine import Mutation

class MyMutation(Mutation):
    def mutate(self, tree, mutation_types):
        # Add your custom mutation type
        mut_type = random.choices(
            list(mutation_types.keys()) + ['my_custom_mutation'],
            weights=list(mutation_types.values()) + [0.1],
            k=1
        )[0]
        
        if mut_type == 'my_custom_mutation':
            return self.my_custom_mutation(tree)
        else:
            return super().mutate(tree, mutation_types)
    
    def my_custom_mutation(self, tree):
        # Your custom mutation logic
        return modified_tree
```

### Does this support regression?

Yes! Set `task_type='regression'`:

```python
initializer = TreeInitializer(
    n_features=n_features,
    n_classes=1,  # Not used for regression
    task_type='regression',
    max_depth=6
)
```

The fitness calculator automatically handles regression (MSE-based).

### Can I use NSGA-II for multi-objective optimization?

Yes! `ParetoOptimizer` in `src/ga_trees/ga/multi_objective.py` implements full
NSGA-II with real crossover and mutation operators:

```python
from ga_trees.ga import ParetoOptimizer

optimizer = ParetoOptimizer(
    initializer=initializer,
    fitness_fn=fitness_calc.calculate_fitness,  # mode='pareto' returns (acc, interp)
    mutation_fn=mutation.mutate,
    random_state=42,
)
front = optimizer.evolve_pareto_front(X_train, y_train, population_size=100, n_generations=50)
```

See `scripts/run_pareto_optimization.py` for a complete example.

### How do I contribute?

See [CONTRIBUTING.md](../../CONTRIBUTING.md) for:
- Development setup
- Code style guidelines
- Testing requirements
- Pull request process

## Troubleshooting

### "RuntimeError: maximum recursion depth exceeded"

Your tree is too deep. Reduce max_depth:

```yaml
tree:
  max_depth: 4  # Instead of 10
```

### "ValueError: tree validation failed"

Check constraint violations:

```python
valid, errors = tree.validate()
print(errors)
```

Common fixes:
- Reduce `max_depth`
- Increase `min_samples_split`
- Check feature ranges

### "Memory Error"

Reduce population size:

```yaml
ga:
  population_size: 50  # Instead of 200
```

### Training hangs or is very slow

1. Check if stuck in validation:
   - Reduce `max_depth`
   - Increase `min_samples_leaf`

2. Reduce complexity:
   - Smaller population
   - Fewer generations
   - Fewer CV folds

3. Check data size:
   - Large datasets take longer
   - Consider sampling for development

### "Import Error: No module named 'ga_trees'"

Run from project root:

```bash
cd ga-optimized-trees
python -m scripts.train --config configs/paper.yaml
```

Or reinstall:

```bash
pip install -e .
```

## Performance

### What's the recommended hardware?

**Minimum:**
- CPU: 2 cores
- RAM: 4 GB
- Disk: 500 MB

**Recommended:**
- CPU: 4+ cores
- RAM: 8+ GB
- Disk: 1 GB

### Can I parallelize training?

Fitness evaluation can be parallelized (planned feature). Currently, training is sequential but efficient for moderate population sizes.

### How much disk space do results need?

**Per experiment:**
- Results CSV: ~10 KB
- Saved models: ~100 KB each
- Figures: ~500 KB each

**Full benchmark suite:**
- ~5 MB total

## Getting Help

### Where can I get help?

1. **Documentation:** Check `docs/` folder
2. **Issues:** [GitHub Issues](https://github.com/ibrah5em/ga-optimized-trees/issues)
3. **Discussions:** [GitHub Discussions](https://github.com/ibrah5em/ga-optimized-trees/discussions)
4. **Email:** ibrah5em@github.com

### How do I report a bug?

Open an issue with:
1. **Environment:**
   ```bash
   python --version
   pip list | grep -E "(ga-trees|numpy|scikit-learn)"
   ```

2. **Steps to reproduce**
3. **Expected vs actual behavior**
4. **Error logs** (if any)

### How do I request a feature?

Open an issue describing:
1. **Use case:** What problem does this solve?
2. **Proposed solution:** How should it work?
3. **Alternatives:** Other approaches you've considered
4. **Additional context:** Any other relevant information

## Research & Publications

### Can I use this for my research?

Yes! The framework is MIT licensed. Please cite:

```bibtex
@software{ga_optimized_trees,
  title={GA-Optimized Decision Trees: Multi-Objective Evolution for Interpretable Machine Learning},
  author={Your Research Team},
  year={2025},
  url={https://github.com/ibrah5em/ga-optimized-trees}
}
```

### Where can I find the research methodology?

See [Methodology](../research/methodology.md) for:
- Experimental design
- Statistical testing approach
- Baseline configurations
- Evaluation metrics

### Are there published papers using this?

Check [Publications](../research/publications.md) for list of papers using this framework.

---

**Still have questions?** Check:
- [Troubleshooting Guide](troubleshooting.md)
- [Performance Tips](performance.md)
- [GitHub Discussions](https://github.com/ibrah5em/ga-optimized-trees/discussions)
