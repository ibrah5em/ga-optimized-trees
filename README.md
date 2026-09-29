# 🌳 GA-Optimized Decision Trees

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Contributions Welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg)](CONTRIBUTING.md)

**A genetic algorithm framework for evolving decision trees that trade accuracy against tree size.**

Instead of growing one tree greedily and pruning it, the GA searches over whole trees, so
the objective can be any property of the tree (size, path length, features used) and the
multi-objective mode returns a whole accuracy–size frontier from one run.

> **Benchmark outcome (pre-registered, 20 OpenML-CC18 datasets).** Evolution beats
> random search over the same tree space at an exactly matched evaluation budget. It does
> **not** beat CART's cost-complexity pruning path, and a tuned GA tree is **not**
> equivalent in accuracy to tuned CART: it loses more than 2 points on 8 of 20 datasets,
> with trees about a third the size. Earlier claims of "46–82% smaller trees at equivalent
> accuracy" were withdrawn; they were not produced by any run. See
> [`paper/STATUS.md`](paper/STATUS.md) for the results,
> [`paper/PREREGISTRATION.md`](paper/PREREGISTRATION.md) for the protocol and verdicts, and
> [`paper/CLAIMS.md`](paper/CLAIMS.md) for the audit.

______________________________________________________________________

## Quick Start

### Installation

```bash
git clone https://github.com/ibrah5em/ga-optimized-trees.git
cd ga-optimized-trees
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

pip install -e .           # Core only
pip install -e .[all]      # All optional features
pip install -e .[dev]      # Development (tests, linting)
```

### Train a Tree

```bash
python scripts/train.py --config configs/paper.yaml --dataset iris
```

### Run Benchmarks

```bash
python scripts/experiment.py --config configs/paper.yaml
```

### Python API

```python
import numpy as np
from ga_trees import GAEngine, GAConfig, TreeInitializer, FitnessCalculator, Mutation
from ga_trees.data import DatasetLoader
from ga_trees.fitness import TreePredictor

# Load data
data = DatasetLoader().load_dataset("iris", test_size=0.2)
X_train, y_train = data["X_train"], data["y_train"]
n_features = X_train.shape[1]
n_classes = len(np.unique(y_train))

# Configure
config = GAConfig(population_size=80, n_generations=40)
initializer = TreeInitializer(
    n_features, n_classes, max_depth=6, min_samples_split=8, min_samples_leaf=3
)
fitness_calc = FitnessCalculator(accuracy_weight=0.68, interpretability_weight=0.32)
feature_ranges = {
    i: (X_train[:, i].min(), X_train[:, i].max()) for i in range(n_features)
}
mutation = Mutation(n_features, feature_ranges)

# Evolve
engine = GAEngine(config, initializer, fitness_calc.calculate_fitness, mutation)
best_tree = engine.evolve(X_train, y_train, verbose=True)

# Predict
predictor = TreePredictor()
y_pred = predictor.predict(best_tree, data["X_test"])
print(f"Tree: {best_tree.get_num_nodes()} nodes, depth {best_tree.get_depth()}")
```

______________________________________________________________________

## Benchmark Results

Pre-registered protocol (`paper/PREREGISTRATION.md`): 20 OpenML-CC18 datasets, nested
cross-validation, a random-search baseline matched to the GA's exact evaluation count,
and CART's full cost-complexity pruning path as the comparator.

| Question                                                     | Answer                                                        |
| ------------------------------------------------------------ | ------------------------------------------------------------- |
| Does evolution beat random search over the same tree space?  | **Yes** — hypervolume +0.66, Holm p = 0.032, 15/20 datasets   |
| Does the GA's frontier beat CART's pruning path?             | **No** — larger hypervolume on 9/20 datasets (45%)            |
| Is a tuned GA tree as accurate as tuned CART (±2 points)?    | **No** — mean −3.9 points; loses > 2 points on 8/20 datasets |
| Are the GA's trees smaller?                                  | Yes — 6.3 leaves vs 18.4, at the accuracy cost above          |

The GA loses on problems where accuracy keeps rising with tree size (vowel, vehicle,
eucalyptus, tic-tac-toe, …): its fronts stop at small trees. Run data for every number is
committed under [`paper/evidence/`](paper/evidence/); details in
[`paper/STATUS.md`](paper/STATUS.md).

Earlier versions of this README claimed "46–82% smaller trees with statistically equivalent
accuracy". Those figures were typed into a plotting script rather than produced by a run,
and the "equivalence" was a non-significant test on dependent folds.
[`paper/CLAIMS.md`](paper/CLAIMS.md) records the audit.

______________________________________________________________________

## How It Works

The GA evolves a population of decision trees using a weighted fitness function:

```
Fitness = w₁ × Accuracy + w₂ × Interpretability
```

In the weighted-sum mode, the interpretability term is a composite of node complexity, feature coherence, tree balance and semantic coherence. It is a **search heuristic only**: results are reported with node count, leaf count, mean decision-path length and distinct features used (see [docs/core-concepts/interpretability.md](docs/core-concepts/interpretability.md)). The evolutionary loop applies tournament selection, subtree crossover with parent tracking, and four mutation operators (threshold perturbation, feature replacement, subtree pruning, leaf expansion).

______________________________________________________________________

## Configuration

Experiments are driven by YAML config files in `configs/`:

| Config                          | Use Case                                              |
| ------------------------------- | ----------------------------------------------------- |
| `paper.yaml`                    | Larger population/generation budget for research runs |
| `default.yaml`                  | General-purpose defaults                              |
| `fast.yaml`                     | Quick experiments (small population, few generations) |
| `balanced.yaml`                 | Equal accuracy/interpretability weight                |
| `accuracy_focused.yaml`         | Maximize accuracy                                     |
| `interpretability_focused.yaml` | Maximize interpretability                             |
| `optimized.yaml`                | Optuna-tuned hyperparameters                          |

```bash
python scripts/train.py --config configs/paper.yaml --dataset breast_cancer
python scripts/experiment.py --config configs/fast.yaml
```

______________________________________________________________________

## Project Structure

```
ga-optimized-trees/
├── src/ga_trees/             # Core package
│   ├── genotype/             # Tree representation (Node, TreeGenotype)
│   ├── ga/                   # GA engine, selection, crossover, mutation
│   ├── fitness/              # Fitness calculation, interpretability metrics
│   ├── baselines/            # CART, Random Forest, XGBoost baselines
│   ├── data/                 # Dataset loading (sklearn, OpenML, CSV)
│   └── evaluation/           # Metrics, visualization, explainability
├── scripts/                  # CLI tools (train, experiment, visualize)
├── configs/                  # YAML configuration files
├── tests/                    # Unit and integration tests
│   ├── unit/                 # Component tests
│   ├── integration/          # End-to-end tests
│   └── test_smoke.py         # Import and workflow smoke tests
├── docs/                     # Documentation
├── notebooks/                # Jupyter notebooks (quick start, EDA)
├── models/                   # Trained model output (gitignored)
└── results/                  # Experiment output
```

______________________________________________________________________

## Testing

```bash
pytest tests/ -v                                    # All tests
pytest tests/unit/ -v                               # Unit tests only
pytest tests/ -v --cov=src/ga_trees                 # With coverage
```

______________________________________________________________________

## Contributing

```bash
pip install -e .[dev]
pre-commit install
pytest tests/ -v
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for full guidelines.

______________________________________________________________________

## Documentation

Full documentation is in [`docs/`](docs/), covering installation, core concepts, API reference, user guides, and research methodology.

Also available at **[ibrah5em.github.io/ga-optimized-trees](https://ibrah5em.github.io/ga-optimized-trees/)**.

______________________________________________________________________

## License

MIT License — see [LICENSE](LICENSE).

Copyright (c) 2025 Ibrahem Hasaki and LuF8y

## Acknowledgments

Built with [DEAP](https://github.com/DEAP/deap), [scikit-learn](https://scikit-learn.org/), [Matplotlib](https://matplotlib.org/), and [Seaborn](https://seaborn.pydata.org/). Thanks to Leen Khalil and Yousef Deeb for their support.
