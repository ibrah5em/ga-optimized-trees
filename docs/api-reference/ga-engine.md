# GA Engine API Reference

Complete documentation for genetic algorithm components.

## Module: `ga_trees.ga.engine`

### GAConfig

Configuration dataclass for genetic algorithm.

**Attributes:**

- `population_size` (int): Number of individuals per generation (default: 100)
- `n_generations` (int): Number of evolution cycles (default: 50)
- `crossover_prob` (float): Crossover probability \[0, 1\] (default: 0.7)
- `mutation_prob` (float): Mutation probability \[0, 1\] (default: 0.2)
- `tournament_size` (int): Tournament selection size (default: 3)
- `elitism_ratio` (float): Fraction of elite preserved \[0, 1\] (default: 0.1)
- `mutation_types` (dict): Mutation operator probabilities (must sum to 1.0). The operator
  is drawn by position, so the key order matters for reproducing a run.
- `random_state` (int | None): Seeds `random` and `numpy.random` at the start of `evolve`
  (default: None)
- `early_stopping_rounds` (int | None): Stop after this many generations without the best
  fitness improving by more than `early_stopping_tol`; None disables it (default: None)
- `early_stopping_tol` (float): Minimum improvement that resets the counter (default: 1e-6)

**Example:**

```python
from ga_trees.ga.engine import GAConfig

config = GAConfig(
    population_size=80,
    n_generations=40,
    crossover_prob=0.72,
    mutation_prob=0.18,
    tournament_size=4,
    elitism_ratio=0.12,
    mutation_types={
        "threshold_perturbation": 0.45,
        "feature_replacement": 0.25,
        "prune_subtree": 0.25,
        "expand_leaf": 0.05,
    },
)
```

______________________________________________________________________

### TreeInitializer

Initialize random decision trees.

#### Constructor

```python
TreeInitializer(
    n_features,
    n_classes,
    max_depth,
    min_samples_split,
    min_samples_leaf,
    task_type="classification",
    growth_stop_prob=0.3,
    split_strategy="midpoint",
)
```

**Parameters:**

- `n_features` (int): Number of input features
- `n_classes` (int): Number of target classes
- `max_depth` (int): Maximum tree depth
- `min_samples_split` (int): Minimum samples to split
- `min_samples_leaf` (int): Minimum samples in leaf
- `task_type` (str): 'classification' or 'regression'
- `growth_stop_prob` (float): Per-node probability of stopping growth when seeding the
  population, in \[0, 1). Lower values seed bushier trees; defaults to
  `DEFAULT_GROWTH_STOP_PROB` (0.3)
- `split_strategy` (str): Where split thresholds come from. `"midpoint"` (default) picks a
  midpoint between observed values of the samples reaching the node, keeping
  `min_samples_leaf` on both sides. `"uniform"` is the original draw across the feature's
  whole range, which often produces splits that send every sample one way.

#### Methods

##### `create_random_tree(X, y)`

Create a random valid tree.

**Parameters:**

- `X` (np.ndarray): Training features
- `y` (np.ndarray): Training labels

**Returns:**

- `TreeGenotype`: Random tree respecting constraints

**Example:**

```python
from ga_trees.ga.engine import TreeInitializer
import numpy as np

X = np.random.rand(100, 4)
y = np.random.randint(0, 2, 100)

initializer = TreeInitializer(
    n_features=4, n_classes=2, max_depth=5, min_samples_split=10, min_samples_leaf=5
)

tree = initializer.create_random_tree(X, y)
print(f"Created tree: depth={tree.get_depth()}, nodes={tree.get_num_nodes()}")
```

______________________________________________________________________

### Mutation

The four mutation operators.

#### Constructor

```python
Mutation(
    n_features, feature_ranges, X=None, min_samples_leaf=1, split_strategy="midpoint"
)
```

**Parameters:**

- `n_features` (int): Number of input features
- `feature_ranges` (dict): `{feature_index: (min, max)}`, used when no `X` is given
- `X` (np.ndarray | None): The data the GA trains on. With it, `threshold_perturbation`,
  `feature_replacement` and `expand_leaf` draw thresholds from the values that actually reach
  the node being mutated. Without it they fall back to `feature_ranges`. Pass the GA's
  training split only, never the validation split, or the search sees the data it's scored on.
- `min_samples_leaf` (int): Minimum samples a new split must leave on each side
- `split_strategy` (str): `"midpoint"` or `"uniform"`, as for `TreeInitializer`

`mutate(tree, mutation_types)` picks one operator by the given probabilities and applies it
in place.

______________________________________________________________________

### GAEngine

Main genetic algorithm engine.

#### Constructor

```python
GAEngine(config, initializer, fitness_function, mutation, repair=None)
```

**Parameters:**

- `config` (GAConfig): GA configuration
- `initializer` (TreeInitializer): Tree initializer
- `fitness_function` (callable): `f(tree, X, y)` returning a float. If you pass a validation
  set to `evolve`, it's called as `f(tree, X, y, X_val, y_val)` instead;
  `FitnessCalculator.calculate_fitness` accepts both.
- `mutation` (Mutation): Mutation operator
- `repair` (callable | None): Applied to every offspring after crossover and mutation. Without
  it, `min_samples_split` / `min_samples_leaf` are only enforced when the initial population
  is created. `ga_trees.ga.repair.repair_from_config(tree_config, X, y)` builds one that
  collapses splits the constraints wouldn't allow.

#### Methods

##### `evolve(X, y, verbose=True, X_val=None, y_val=None)`

Run the evolution process.

**Parameters:**

- `X` (np.ndarray): Training features; leaf predictions are fitted on these
- `y` (np.ndarray): Training labels
- `verbose` (bool): Log progress
- `X_val`, `y_val` (np.ndarray | None): Held-out data to score fitness on. Pass both or
  neither. Without them fitness is resubstitution (leaves fitted and scored on the same
  rows), which rewards whichever tree overfits hardest.

**Returns:**

- `TreeGenotype`: Best individual found, by validation fitness when a validation set is given

**Example:**

```python
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

from ga_trees.fitness.calculator import FitnessCalculator
from ga_trees.ga.engine import GAConfig, GAEngine, Mutation, TreeInitializer

X, y = load_breast_cancer(return_X_y=True)
X_fit, X_val, y_fit, y_val = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=0
)
n_features = X.shape[1]

ga_engine = GAEngine(
    config=GAConfig(population_size=60, n_generations=30, random_state=42),
    initializer=TreeInitializer(
        n_features=n_features,
        n_classes=2,
        max_depth=5,
        min_samples_split=10,
        min_samples_leaf=5,
    ),
    fitness_function=FitnessCalculator(
        accuracy_weight=0.9, interpretability_weight=0.1
    ).calculate_fitness,
    mutation=Mutation(
        n_features=n_features,
        feature_ranges={
            i: (X_fit[:, i].min(), X_fit[:, i].max()) for i in range(n_features)
        },
        X=X_fit,
        min_samples_leaf=5,
    ),
)

best_tree = ga_engine.evolve(X_fit, y_fit, X_val=X_val, y_val=y_val, verbose=False)
print(
    f"Best validation fitness: {best_tree.fitness_:.4f}, {best_tree.get_num_nodes()} nodes"
)
```

##### `get_history()`

Get evolution history.

**Returns:**

- `dict`: History with keys:
  - `best_fitness`: List of best fitness per generation
  - `avg_fitness`: List of average fitness per generation
  - `diversity`: List of population diversity (if tracked)

**Example:**

```python
history = ga_engine.get_history()

import matplotlib.pyplot as plt

plt.plot(history["best_fitness"], label="Best")
plt.plot(history["avg_fitness"], label="Average")
plt.xlabel("Generation")
plt.ylabel("Fitness")
plt.legend()
plt.show()
```

______________________________________________________________________

See [Genotype API](genotype.md) for tree structure documentation.
