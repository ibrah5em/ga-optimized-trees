# Architecture Overview

This document describes the system architecture and design principles of the GA-Optimized Decision Trees framework.

## System Architecture

```
ga-optimized-trees/
│
├── Genotype Layer (Tree Representation)
│   └── TreeGenotype, Node
│
├── GA Engine Layer (Evolution)
│   ├── Initialization
│   ├── Selection
│   ├── Crossover
│   └── Mutation
│
├── Fitness Layer (Evaluation)
│   ├── Accuracy Calculation
│   ├── Interpretability Metrics
│   └── Multi-Objective Combination
│
├── Evaluation Layer (Analysis)
│   ├── Metrics
│   ├── Visualization
│   └── Statistical Tests
│
└── Application Layer (User Interface)
    ├── Training Scripts
    ├── Experiment Runners
    └── Configuration Management
```

## Core Components

### 1. Genotype Module (`src/ga_trees/genotype/`)

**Purpose:** Represent decision trees as evolvable structures.

**Key Classes:**

- `TreeGenotype`: Main tree representation
- `Node`: Individual tree nodes (internal/leaf)

**Responsibilities:**

- Tree structure management
- Constraint validation
- Tree manipulation (copy, prune, expand)
- Rule extraction

**Design Decisions:**

- **Binary trees only** for simplicity
- **Constrained structure** (max depth, min samples)
- **Deep copy support** for safe evolution
- **Validation hooks** for constraint checking

### 2. GA Engine (`src/ga_trees/ga/`)

**Purpose:** Evolve populations of trees using genetic algorithms.

**Key Classes:**

- `GAEngine`: Main evolution loop
- `TreeInitializer`: Create initial population
- `Selection`: Tournament and elitism selection
- `Crossover`: Subtree-aware crossover
- `Mutation`: Four mutation operators

**Evolution Process:**

```
1. Initialize Population
   ↓
2. Evaluate Fitness
   ↓
3. Select Parents (Tournament)
   ↓
4. Apply Crossover (70% prob)
   ↓
5. Apply Mutation (18% prob)
   ↓
6. Evaluate Offspring
   ↓
7. Replace Population (with elitism)
   ↓
8. Repeat 3-7 for N generations
```

**Mutation Operators:**

- **Threshold Perturbation (45%):** Adjust split thresholds
- **Feature Replacement (25%):** Change split features
- **Prune Subtree (25%):** Simplify tree
- **Expand Leaf (5%):** Increase complexity

### 3. Fitness Calculator (`src/ga_trees/fitness/`)

**Purpose:** Evaluate tree quality using multiple objectives.

**Key Classes:**

- `FitnessCalculator`: Main fitness computation
- `TreePredictor`: Make predictions
- `InterpretabilityCalculator`: Measure interpretability

**Fitness Formula:**

```
Fitness = w₁ × Accuracy + w₂ × Interpretability

where:
  Accuracy = sklearn.metrics.accuracy_score()

  Interpretability = Σ wᵢ × ComponentScoreᵢ
  Components:
    - Node Complexity (60%): e^(-nodes/15)
    - Feature Coherence (20%): 1 - (unique_features / internal_nodes)
    - Tree Balance (10%): 1 - (std_depths / max_depth)
    - Semantic Coherence (10%): 1 - (entropy / max_entropy)
```

**Default Weights:**

- Accuracy: 68%
- Interpretability: 32%

### 4. Evaluation Tools (`src/ga_trees/evaluation/`)

**Purpose:** Analyze and visualize results.

**Key Modules:**

- `metrics.py`: Comprehensive metrics
- `tree_visualizer.py`: Graphviz visualization
- `feature_importance.py`: Feature analysis
- `explainability.py`: LIME/SHAP integration (optional)

## Data Flow

```
1. User loads dataset
   ↓
2. Configuration loaded from YAML
   ↓
3. GA Engine initializes population
   ↓
4. For each generation:
   a. Fitness Calculator evaluates all trees
   b. TreePredictor makes predictions
   c. Metrics computed
   d. GA operators create offspring
   ↓
5. Best tree selected
   ↓
6. Evaluation tools analyze results
   ↓
7. Visualization and export
```

## Configuration System

### YAML-Based Configuration

All hyperparameters are externalized to YAML files:

```yaml
ga:
  population_size: 80
  n_generations: 40
  mutation_types:
    threshold_perturbation: 0.45
    feature_replacement: 0.25
    prune_subtree: 0.25
    expand_leaf: 0.05

tree:
  max_depth: 6
  min_samples_split: 8
  min_samples_leaf: 3

fitness:
  weights:
    accuracy: 0.68
    interpretability: 0.32
  interpretability_weights:
    node_complexity: 0.55
    feature_coherence: 0.25
    tree_balance: 0.10
    semantic_coherence: 0.10
```

**Benefits:**

- **Reproducibility:** Same config = same results
- **Experimentation:** Easy to compare settings
- **Version control:** Track configuration changes
- **Sharing:** Share configs with collaborators

## Design Principles

### 1. Modularity

Each component has a single, well-defined responsibility.

### 2. Extensibility

Easy to add:

- New mutation operators
- Custom fitness functions
- Additional datasets
- Alternative algorithms

### 3. Reproducibility

- Deterministic when seeds are set
- Configuration-driven experiments
- Comprehensive logging

### 4. Performance

- Vectorized numpy operations
- Parallel fitness evaluation (optional)
- Efficient tree operations

### 5. Research-Oriented

- Statistical rigor (20-fold CV)
- Baseline comparisons
- Publication-quality visualizations
- Detailed metrics

## Key Algorithms

### Subtree Crossover

```python
def subtree_crossover(parent1, parent2):
    child1 = parent1.copy()
    child2 = parent2.copy()

    # Select random nodes
    node1 = random.choice(child1.get_all_nodes()[1:])
    node2 = random.choice(child2.get_all_nodes()[1:])

    # Swap subtrees
    swap_node_contents(node1, node2)

    # Repair if needed
    child1 = repair_tree(child1)
    child2 = repair_tree(child2)

    return child1, child2
```

### Tournament Selection

```python
def tournament_selection(population, tournament_size, n_select):
    selected = []
    for _ in range(n_select):
        tournament = random.sample(population, tournament_size)
        winner = max(tournament, key=lambda t: t.fitness_)
        selected.append(winner.copy())
    return selected
```

## Constraint Handling

Trees must satisfy:

1. **Max Depth:** `depth ≤ max_depth`
1. **Min Samples Split:** `samples ≥ min_samples_split`
1. **Min Samples Leaf:** `leaf_samples ≥ min_samples_leaf`
1. **Valid Features:** `0 ≤ feature_idx < n_features`

**Enforcement:**

- **Initialization:** Only create valid trees
- **Crossover:** Repair after swapping
- **Mutation:** Prune if constraints violated
- **Validation:** Check before fitness evaluation

## Performance Considerations

### Time Complexity

- **Initialization:** O(pop_size × tree_size)
- **Fitness Evaluation:** O(pop_size × n_samples × tree_depth)
- **Selection:** O(pop_size × tournament_size)
- **Crossover:** O(tree_size)
- **Mutation:** O(tree_size)

### Space Complexity

- **Population:** O(pop_size × tree_size)
- **Data:** O(n_samples × n_features)

### Typical Runtime

- **Iris (150 samples):** ~3 seconds/fold
- **Wine (178 samples):** ~5 seconds/fold
- **Breast Cancer (569 samples):** ~7 seconds/fold

Full 20-fold CV benchmark: ~17 minutes

## Extension Points

### Add Custom Mutation

```python
class CustomMutation(Mutation):
    def your_mutation(self, tree):
        # Your logic here
        return modified_tree
```

### Add Custom Fitness

```python
class CustomFitness(FitnessCalculator):
    def calculate_fitness(self, tree, X, y):
        # Your logic here
        return fitness_score
```

### Add Custom Metric

```python
def your_metric(tree):
    # Compute custom interpretability metric
    return score
```

## Related Documents

- [Genetic Algorithm Details](genetic-algorithm.md)
- [Tree Representation](tree-representation.md)
- [Fitness Functions](fitness-functions.md)
- [API Reference](../api-reference/)
