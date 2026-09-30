# 🌳 GA-Optimized Decision Trees

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Contributions Welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg)](CONTRIBUTING.md)

A genetic algorithm that evolves decision trees, trading accuracy against tree size.

CART grows a tree one greedy split at a time and prunes it afterwards. This project searches
over whole trees instead, so the objective can be anything you can compute from a tree:
node count, features used, path length, a custom metric. The multi-objective mode (NSGA-II)
hands back an accuracy–size frontier from a single run, and you pick the point you want.

## Does it work?

Partly. We pre-registered a benchmark on 20 OpenML-CC18 datasets and wrote down in advance
what would count as failure. Two of the three tests failed.

| Question                                                    | Answer                                                       |
| ----------------------------------------------------------- | ------------------------------------------------------------ |
| Does evolution beat random search over the same tree space? | **Yes** — hypervolume +0.66, Holm p = 0.032, 15/20 datasets  |
| Does the GA's frontier beat CART's pruning path?            | **No** — larger hypervolume on 9/20 datasets (45%)           |
| Is a tuned GA tree as accurate as tuned CART (±2 points)?   | **No** — mean −3.9 points; loses > 2 points on 8/20 datasets |
| Are the GA's trees smaller?                                 | Yes — 6.3 leaves vs 18.4, at the accuracy cost above         |

So the search itself does its job: with exactly the same number of tree evaluations, it
finds better trees than random sampling does. What it doesn't do is beat CART, which has a
40-year head start on this exact problem.

![GA minus tuned CART, test accuracy per dataset with 90% intervals](docs/assets/figures/fig_k3.png)

The losses aren't random. They land on datasets where accuracy keeps climbing as the tree
gets bigger (vowel, eucalyptus, tic-tac-toe, vehicle). The GA's fronts stop at a median of
5.4 nodes while CART's run to 32, so on those problems CART simply reaches further:

![Test-fold frontiers on breast-w and vehicle](docs/assets/figures/fig_frontiers.png)

On breast-w, where a handful of nodes is enough, the GA's frontier sits at or above CART's.
On vehicle, CART's larger trees pull ahead. An ablation on the four worst datasets explains
only part of this truncation, and the rest is still open. [`paper/STATUS.md`](paper/STATUS.md)
has the full story.

> **A note on the old numbers.** Earlier versions of this README claimed "46–82% smaller
> trees with statistically equivalent accuracy". Those figures were typed into a plotting
> script, not produced by any run, and the "equivalence" was a non-significant t-test on
> dependent folds. They're withdrawn. [`paper/CLAIMS.md`](paper/CLAIMS.md) is the audit, and
> everything above is regenerated from the run data in [`paper/evidence/`](paper/evidence/).

## When to use it

Use **CART** if you want the most accurate small tree for plain accuracy. It's faster and,
on our benchmark, better.

Use the **GA** when the thing you're optimising isn't something a greedy split rule can
target: a cap on distinct features, a feature-cost budget, a custom metric over the whole
tree, or when you want to see the whole accuracy–size frontier and choose from it.

## Quick start

```bash
git clone https://github.com/ibrah5em/ga-optimized-trees.git
cd ga-optimized-trees
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

pip install -e .           # core only
pip install -e .[all]      # all optional features
pip install -e .[dev]      # tests, linting
```

Train a tree from a config:

```bash
python scripts/train.py --config configs/default.yaml --dataset breast_cancer
```

Or from Python. This takes a few seconds and prints something like
`11 nodes, depth 5, test accuracy 0.939`:

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

Two things in there matter more than they look. Passing `X_val` stops the search from
rewarding memorisation. And keep `interpretability_weight` small: at 0.35, a tree has to gain
about 23 accuracy points before growing from a stump to CART's size pays off, so the GA
settles on stumps and looks worse than it is.

## How it works

Each individual is a complete binary tree. Every generation the GA:

1. **Selects** parents by tournament.
1. **Crosses them over** by swapping subtrees, then checks depth and sample constraints.
1. **Mutates** with one of four operators: nudge a threshold, swap a split's feature, prune
   a subtree to a leaf, or grow a leaf into a split. Given the training data, thresholds are
   drawn from values that actually reach the node, so a mutation always changes the split.
1. **Scores** the offspring: `w × accuracy + (1 − w) × interpretability`, or both objectives
   separately under NSGA-II.

The "interpretability" term is a composite of node count, feature reuse, balance and path
consistency. It steers the search and nothing more: it isn't a validated measure of how
understandable a tree is, so results are reported as node count, leaves, path length and
features used ([why](docs/core-concepts/interpretability.md)).

## Configs

Every script takes `--config`:

| Config                          | What it's for                                               |
| ------------------------------- | ----------------------------------------------------------- |
| `paper.yaml`                    | The pre-registered benchmark ran with this. Don't edit it.  |
| `paper-repair.yaml`             | Same, with constraint repair on (the sensitivity run)       |
| `default.yaml`                  | General-purpose starting point                              |
| `fast.yaml`                     | Small population, few generations, for quick iteration      |
| `balanced.yaml`                 | Equal weight on accuracy and the size heuristic             |
| `accuracy_focused.yaml`         | Weight mostly on accuracy, bigger budget                    |
| `interpretability_focused.yaml` | Weight mostly on the size heuristic; expect stumps          |
| `optimized.yaml`                | GA settings from an earlier Optuna search, not re-validated |

## Reproducing the paper

Every number above comes from a committed run. Each folder in
[`paper/evidence/`](paper/evidence/) has the fold-level CSVs, the seeds, and the exact
command that produced them. The main two:

```bash
python scripts/frontier_benchmark.py --config configs/paper.yaml --n-jobs 5   # K1, K2 (hours)
python scripts/benchmark.py --config configs/paper.yaml --outer-repeats 1 --no-depth-tuning --n-jobs 4   # K3
python scripts/paper_assets.py   # regenerate every table, figure and number in the papers
```

The write-ups:

- [`paper/general/`](paper/general/): the full study as a long-form article
- [`paper/gecco/`](paper/gecco/): the conference version
- [`paper/PREREGISTRATION.md`](paper/PREREGISTRATION.md): hypotheses and kill criteria, fixed
  before the runs
- [`paper/STATUS.md`](paper/STATUS.md): the one-page summary

## Project layout

```
src/ga_trees/
├── genotype/     tree representation
├── ga/           engine, operators, NSGA-II, split points, constraint repair
├── fitness/      prediction and the fitness function
├── benchmark/    nested CV, frontiers, baselines for the pre-registered runs
├── evaluation/   hypervolume, statistics, figures, metrics, visualisation
├── baselines/    CART, random forest, XGBoost wrappers
└── data/         dataset loading (scikit-learn, OpenML, CSV)
scripts/          CLI entry points: train, benchmark, paper assets
configs/          YAML configs
paper/            pre-registration, evidence, both papers
tests/            unit and integration tests
docs/             documentation site source
```

## Tests

```bash
pytest tests/ -v
pytest tests/unit/ --cov=src/ga_trees --cov-fail-under=80
```

## Contributing

```bash
pip install -e .[dev]
pre-commit install
pytest tests/ -v
```

See [CONTRIBUTING.md](CONTRIBUTING.md). The docs site is
**[ibrah5em.github.io/ga-optimized-trees](https://ibrah5em.github.io/ga-optimized-trees/)**.

## License

MIT, see [LICENSE](LICENSE). Copyright (c) 2025 Ibrahem Hasaki and LuF8y.

Built on [DEAP](https://github.com/DEAP/deap), [scikit-learn](https://scikit-learn.org/),
[Matplotlib](https://matplotlib.org/) and [Seaborn](https://seaborn.pydata.org/). Thanks to
Leen Khalil and Yousef Deeb for their support.
