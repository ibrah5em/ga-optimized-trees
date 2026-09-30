# GA-Optimized Decision Trees

<div class="hero-banner" markdown>

# 🌳 GA-Optimized Decision Trees

**Evolving decision trees that balance accuracy and interpretability using multi-objective genetic algorithms.**

Search the accuracy–complexity spectrum directly and choose the operating point, instead of accepting whatever a greedy split rule produces.

</div>

<div class="stats-row" markdown>
<div class="stat-card" markdown>
<div class="stat-num">NSGA‑II</div>
<div class="stat-desc">Multi-objective search</div>
</div>
<div class="stat-card" markdown>
<div class="stat-num">4</div>
<div class="stat-desc">Mutation operators</div>
</div>
<div class="stat-card" markdown>
<div class="stat-num">25+</div>
<div class="stat-desc">Loadable datasets</div>
</div>
<div class="stat-card" markdown>
<div class="stat-num">3.8–3.12</div>
<div class="stat-desc">Python support</div>
</div>
</div>

> **Where this stands.** The pre-registered benchmark on 20 OpenML-CC18 datasets is done.
> Evolution beats random search over the same tree space at a matched budget, but it does
> **not** beat CART's cost-complexity pruning path, and a tuned GA tree is **not** as
> accurate as tuned CART. Earlier "46–82% smaller trees at equivalent accuracy" claims on
> this site were withdrawn. See [Results](research/results.md).

______________________________________________________________________

## 🚀 Quick Start

Install:

```bash
git clone https://github.com/ibrah5em/ga-optimized-trees.git
cd ga-optimized-trees
python -m venv venv && source venv/bin/activate
pip install -e .          # core only
pip install -e .[all]     # all optional features
```

Train a tree from a config:

```bash
python scripts/train.py --config configs/paper.yaml --dataset iris
```

Or from Python:

```python
import numpy as np
from sklearn.model_selection import train_test_split

from ga_trees import FitnessCalculator, GAConfig, GAEngine, Mutation, TreeInitializer
from ga_trees.data import DatasetLoader
from ga_trees.fitness import TreePredictor

data = DatasetLoader().load_dataset("breast_cancer", test_size=0.2)
X_train, y_train = data["X_train"], data["y_train"]

# Hold a slice of the training data out of the search. Without it the GA fits
# and scores leaves on the same rows and rewards whichever tree overfits most.
X_fit, X_val, y_fit, y_val = train_test_split(
    X_train, y_train, test_size=0.25, stratify=y_train, random_state=0
)
n_features = X_fit.shape[1]

initializer = TreeInitializer(
    n_features, n_classes=2, max_depth=5, min_samples_split=10, min_samples_leaf=5
)
mutation = Mutation(
    n_features,
    feature_ranges={
        i: (X_fit[:, i].min(), X_fit[:, i].max()) for i in range(n_features)
    },
    X=X_fit,  # lets mutation pick thresholds from values that actually reach the node
    min_samples_leaf=5,
)
fitness = FitnessCalculator(accuracy_weight=0.9, interpretability_weight=0.1)

engine = GAEngine(
    GAConfig(population_size=80, n_generations=40, random_state=42),
    initializer,
    fitness.calculate_fitness,
    mutation,
)
best = engine.evolve(X_fit, y_fit, X_val=X_val, y_val=y_val, verbose=False)

# Structure was chosen on the validation split; refit the leaves on all training rows.
predictor = TreePredictor()
predictor.fit_leaf_predictions(best, X_train, y_train)
accuracy = np.mean(predictor.predict(best, data["X_test"]) == data["y_test"])
print(
    f"{best.get_num_nodes()} nodes, depth {best.get_depth()}, test accuracy {accuracy:.3f}"
)
```

The pre-registered benchmark itself is `scripts/frontier_benchmark.py` (frontiers, K1/K2)
and `scripts/benchmark.py` (single tuned tree, K3); `paper/evidence/` has the configs and
commands for every committed run.

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

| Question                                                    | Answer                                                       |
| ----------------------------------------------------------- | ------------------------------------------------------------ |
| Does evolution beat random search over the same tree space? | **Yes** — hypervolume +0.66, Holm p = 0.032, 15/20 datasets  |
| Does the GA's frontier beat CART's pruning path?            | **No** — larger hypervolume on 9/20 datasets (45%)           |
| Is a tuned GA tree as accurate as tuned CART (±2 points)?   | **No** — mean −3.9 points; loses > 2 points on 8/20 datasets |
| Are the GA's trees smaller?                                 | Yes — 6.3 leaves vs 18.4, at the accuracy cost above         |

Details, per-dataset numbers and the ablations are on the [Results](research/results.md)
page and in `paper/STATUS.md`.

______________________________________________________________________

## 🆚 When to use it

| Aspect    | CART                    | GA-Optimized                                                   |
| --------- | ----------------------- | -------------------------------------------------------------- |
| Search    | Greedy, top-down        | Evolutionary, over whole trees                                 |
| Objective | Impurity, then pruning  | Any function of the tree — size, features used, custom metrics |
| Speed     | Milliseconds            | Seconds to minutes                                             |
| Accuracy  | Better on our benchmark | Behind CART where accuracy keeps rising with tree size         |

Use CART when you want the most accurate small tree for plain accuracy. The GA is worth
it when the objective is something a greedy split rule can't optimise directly.

______________________________________________________________________

## 🔗 Links

<span class="badge badge-slate">v1.0.0</span>
<span class="badge badge-blue">Python 3.8+</span>
<span class="badge badge-muted">MIT License</span>

- **GitHub**: [ibrah5em/ga-optimized-trees](https://github.com/ibrah5em/ga-optimized-trees)
- **Issues**: [github.com/ibrah5em/ga-optimized-trees/issues](https://github.com/ibrah5em/ga-optimized-trees/issues)
- **Discussions**: [github.com/ibrah5em/ga-optimized-trees/discussions](https://github.com/ibrah5em/ga-optimized-trees/discussions)
