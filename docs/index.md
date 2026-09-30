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

To compare the GA with CART and random search on your own data, see the benchmarking
harness: `scripts/benchmark.py` (one tuned tree per method, nested CV) and
`scripts/frontier_benchmark.py` (accuracy–size frontiers).

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
<a href="advanced/statistical-tests/">
<div class="card-icon">📊</div>
<div class="card-title">Evaluating models</div>
<div class="card-desc">Comparing methods across datasets, done properly</div>
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

## 📄 Research paper

A paper on the full study behind this framework is in preparation. It'll be linked here
when it's out; until then, cite the software (see the [FAQ](faq/faq.md)).

______________________________________________________________________

## 🆚 When to use it

| Aspect    | CART                   | GA-Optimized                                                   |
| --------- | ---------------------- | -------------------------------------------------------------- |
| Search    | Greedy, top-down       | Evolutionary, over whole trees                                 |
| Objective | Impurity, then pruning | Any function of the tree — size, features used, custom metrics |
| Speed     | Milliseconds           | Seconds to minutes                                             |

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
