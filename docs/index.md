# GA-Optimized Decision Trees

<div class="hero-banner" markdown>

# 🌳 GA-Optimized Decision Trees

**Evolving decision trees that balance accuracy and interpretability using multi-objective genetic algorithms.**

Achieve **46–82% smaller trees** with statistically equivalent accuracy — validated with 20-fold cross-validation (p > 0.05 on all benchmarks).

</div>

<div class="stats-row" markdown>
<div class="stat-card" markdown>
<div class="stat-num">82%</div>
<div class="stat-desc">Max size reduction</div>
</div>
<div class="stat-card" markdown>
<div class="stat-num">p &gt; 0.05</div>
<div class="stat-desc">Accuracy parity</div>
</div>
<div class="stat-card" markdown>
<div class="stat-num">3</div>
<div class="stat-desc">Benchmark datasets</div>
</div>
<div class="stat-card" markdown>
<div class="stat-num">20‑CV</div>
<div class="stat-desc">Fold validation</div>
</div>
</div>

______________________________________________________________________

## 🚀 Quick Start

=== "Install"

````
```bash
git clone https://github.com/ibrah5em/ga-optimized-trees.git
cd ga-optimized-trees
python -m venv venv && source venv/bin/activate
pip install -e .          # core only
pip install -e .[all]     # all features
```
````

=== "Train a tree"

````
```bash
python scripts/train.py --config configs/paper.yaml --dataset iris
```
````

=== "Python API"

````
```python
from ga_trees import GAEngine, GAConfig, TreeInitializer, FitnessCalculator, Mutation
from ga_trees.data import DatasetLoader
from ga_trees.fitness import TreePredictor
import numpy as np

data = DatasetLoader().load_dataset("iris", test_size=0.2)
X_train, y_train = data["X_train"], data["y_train"]

config   = GAConfig(population_size=80, n_generations=40)
init     = TreeInitializer(X_train.shape[1], len(np.unique(y_train)), max_depth=6)
fitness  = FitnessCalculator(accuracy_weight=0.68, interpretability_weight=0.32)
mutation = Mutation(X_train.shape[1], {i: (X_train[:,i].min(), X_train[:,i].max())
                                        for i in range(X_train.shape[1])})

engine = GAEngine(config, init, fitness.calculate_fitness, mutation)
best   = engine.evolve(X_train, y_train, verbose=True)
print(f"Tree: {best.get_num_nodes()} nodes, depth {best.get_depth()}")
```
````

=== "Run benchmarks"

````
```bash
python scripts/experiment.py --config configs/paper.yaml
```
````

______________________________________________________________________

## 📚 Documentation

<div class="card-grid" markdown>

<div class="card" markdown>
<a href="getting-started/installation/">
<div class="card-icon">⚙️</div>
<div class="card-title">Installation</div>
<div class="card-desc">All install methods, extras, and platform notes</div>
</a>
</div>

<div class="card" markdown>
<a href="getting-started/quickstart/">
<div class="card-icon">⚡</div>
<div class="card-title">Quick Start</div>
<div class="card-desc">Up and running in 5 minutes</div>
</a>
</div>

<div class="card" markdown>
<a href="core-concepts/architecture/">
<div class="card-icon">🏗️</div>
<div class="card-title">Architecture</div>
<div class="card-desc">System design and component overview</div>
</a>
</div>

<div class="card" markdown>
<a href="core-concepts/genetic-algorithm/">
<div class="card-icon">🧬</div>
<div class="card-title">Genetic Algorithm</div>
<div class="card-desc">Selection, crossover, mutation internals</div>
</a>
</div>

<div class="card" markdown>
<a href="api-reference/ga-engine/">
<div class="card-icon">📖</div>
<div class="card-title">API Reference</div>
<div class="card-desc">Full class and method documentation</div>
</a>
</div>

<div class="card" markdown>
<a href="research/results/">
<div class="card-icon">📊</div>
<div class="card-title">Results</div>
<div class="card-desc">Benchmark tables and statistical tests</div>
</a>
</div>

<div class="card" markdown>
<a href="examples/iris/">
<div class="card-icon">🌸</div>
<div class="card-title">Examples</div>
<div class="card-desc">Iris, medical, and credit scoring walkthroughs</div>
</a>
</div>

<div class="card" markdown>
<a href="faq/faq/">
<div class="card-icon">❓</div>
<div class="card-title">FAQ</div>
<div class="card-desc">Common questions and troubleshooting</div>
</a>
</div>

</div>

______________________________________________________________________

## 📈 Benchmark Results

| Dataset           | GA Accuracy    | CART Accuracy  | GA Nodes | CART Nodes | Size Reduction |
| ----------------- | -------------- | -------------- | -------- | ---------- | -------------- |
| **Iris**          | 94.55 ± 8.07%  | 92.41 ± 10.43% | 7.4      | 16.4       | **55%**        |
| **Wine**          | 88.19 ± 10.39% | 87.22 ± 10.70% | 10.7     | 20.7       | **48%**        |
| **Breast Cancer** | 91.05 ± 5.60%  | 91.57 ± 3.92%  | 6.5      | 35.5       | **82%**        |

> All p-values > 0.05 — accuracy difference is not statistically significant.

______________________________________________________________________

## 🆚 How It Compares

| Aspect           | CART           | Random Forest | **GA-Optimized**          |
| ---------------- | -------------- | ------------- | ------------------------- |
| Optimization     | Greedy (local) | Ensemble      | **Global (evolutionary)** |
| Objectives       | Accuracy only  | Accuracy only | **Multi-objective**       |
| Interpretability | No control     | Black box     | **Explicit control ✓**    |
| Tree size        | Often large    | N/A           | **Controllable ✓**        |

______________________________________________________________________

## 🔗 Links

<span class="badge badge-slate">v1.0.0</span>
<span class="badge badge-blue">Python 3.8+</span>
<span class="badge badge-muted">MIT License</span>

- **GitHub**: [ibrah5em/ga-optimized-trees](https://github.com/ibrah5em/ga-optimized-trees)
- **Issues**: [github.com/ibrah5em/ga-optimized-trees/issues](https://github.com/ibrah5em/ga-optimized-trees/issues)
- **Discussions**: [github.com/ibrah5em/ga-optimized-trees/discussions](https://github.com/ibrah5em/ga-optimized-trees/discussions)
