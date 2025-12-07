# GA Engine API Reference

Complete documentation for genetic algorithm components.

## Module: `ga_trees.ga.engine`

### GAConfig

Configuration dataclass for genetic algorithm.

**Attributes:**
- `population_size` (int): Number of individuals per generation (default: 100)
- `n_generations` (int): Number of evolution cycles (default: 50)
- `crossover_prob` (float): Crossover probability [0, 1] (default: 0.7)
- `mutation_prob` (float): Mutation probability [0, 1] (default: 0.2)
- `tournament_size` (int): Tournament selection size (default: 3)
- `elitism_ratio` (float): Fraction of elite preserved [0, 1] (default: 0.1)
- `mutation_types` (dict): Mutation operator probabilities (must sum to 1.0)

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
        'threshold_perturbation': 0.45,
        'feature_replacement': 0.25,
        'prune_subtree': 0.25,
        'expand_leaf': 0.05
    }
)
```

---

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
    task_type='classification'
)
```

**Parameters:**
- `n_features` (int): Number of input features
- `n_classes` (int): Number of target classes
- `max_depth` (int): Maximum tree depth
- `min_samples_split` (int): Minimum samples to split
- `min_samples_leaf` (int): Minimum samples in leaf
- `task_type` (str): 'classification' or 'regression'

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
    n_features=4,
    n_classes=2,
    max_depth=5,
    min_samples_split=10,
    min_samples_leaf=5
)

tree = initializer.create_random_tree(X, y)
print(f"Created tree: depth={tree.get_depth()}, nodes={tree.get_num_nodes()}")
```

---

### GAEngine

Main genetic algorithm engine.

#### Constructor

```python
GAEngine(config, initializer, fitness_function, mutation)
```

**Parameters:**
- `config` (GAConfig): GA configuration
- `initializer` (TreeInitializer): Tree initializer
- `fitness_function` (callable): Function to evaluate fitness
- `mutation` (Mutation): Mutation operator

#### Methods

##### `evolve(X, y, verbose=True)`

Run the evolution process.

**Parameters:**
- `X` (np.ndarray): Training features
- `y` (np.ndarray): Training labels
- `verbose` (bool): Print progress

**Returns:**
- `TreeGenotype`: Best individual found

**Example:**
```python
from ga_trees.ga.engine import GAEngine, GAConfig, TreeInitializer, Mutation
from ga_trees.fitness.calculator import FitnessCalculator

# Setup components
ga_config = GAConfig(population_size=80, n_generations=40)
initializer = TreeInitializer(n_features=4, n_classes=2, max_depth=5,
                              min_samples_split=10, min_samples_leaf=5)
fitness_calc = FitnessCalculator()
mutation = Mutation(n_features=4, feature_ranges={i: (0, 1) for i in range(4)})

# Create engine
ga_engine = GAEngine(
    config=ga_config,
    initializer=initializer,
    fitness_function=fitness_calc.calculate_fitness,
    mutation=mutation
)

# Train
best_tree = ga_engine.evolve(X_train, y_train, verbose=True)
print(f"Best fitness: {best_tree.fitness_:.4f}")
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
plt.plot(history['best_fitness'], label='Best')
plt.plot(history['avg_fitness'], label='Average')
plt.xlabel('Generation')
plt.ylabel('Fitness')
plt.legend()
plt.show()
```

---

See [Genotype API](genotype.md) for tree structure documentation.