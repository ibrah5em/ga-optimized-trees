# Code Style Guide

Standards for code formatting and documentation.

## Python Style

Follow **PEP 8** with these specifics:

### Formatting

**Line Length:** 100 characters

```python
# Good
def calculate_fitness_score(tree, X, y, weights):
    pass


# Bad (> 100 chars)
def calculate_fitness_score_with_interpretability_and_accuracy_weights(
    tree, X, y, weights
):
    pass
```

**Indentation:** 4 spaces

```python
# Good
if condition:
    do_something()

# Bad
if condition:
    do_something()  # 2 spaces
```

### Naming Conventions

```python
# Classes: CamelCase
class TreeGenotype:
    pass


# Functions/methods: snake_case
def calculate_fitness(tree, X, y):
    pass


# Variables: snake_case
population_size = 100

# Constants: UPPER_SNAKE_CASE
MAX_DEPTH = 10


# Private: _leading_underscore
def _internal_helper():
    pass
```

### Imports

Order and group imports:

```python
# 1. Standard library
import os
import sys
from typing import List, Dict

# 2. Third-party
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

# 3. Local
from ga_trees.genotype.tree_genotype import TreeGenotype
from ga_trees.fitness.calculator import FitnessCalculator
```

### Docstrings

Use **Google style**:

```python
def evaluate_tree(tree: TreeGenotype, X: np.ndarray, y: np.ndarray) -> float:
    """Evaluate tree performance on dataset.

    This function fits leaf predictions and calculates accuracy.

    Args:
        tree: Tree genotype to evaluate
        X: Feature matrix of shape (n_samples, n_features)
        y: Target vector of shape (n_samples,)

    Returns:
        Accuracy score between 0 and 1

    Raises:
        ValueError: If tree is invalid or data shapes don't match

    Example:
        >>> tree = create_sample_tree()
        >>> acc = evaluate_tree(tree, X_train, y_train)
        >>> print(f"Accuracy: {acc:.4f}")
        Accuracy: 0.9234
    """
    # Implementation
```

### Type Hints

Always use type hints:

```python
from typing import List, Tuple, Optional, Union


def crossover(
    parent1: TreeGenotype, parent2: TreeGenotype
) -> Tuple[TreeGenotype, TreeGenotype]:
    """Perform crossover."""
    pass


def mutate(tree: TreeGenotype, rate: float = 0.2) -> Optional[TreeGenotype]:
    """Mutate tree."""
    pass
```

## Tools

### Black

Auto-formatter:

```bash
black src/ tests/ scripts/ --line-length 100
```

### isort

Import sorting:

```bash
isort src/ tests/ scripts/ --profile black --line-length 100
```

### flake8

Linting:

```bash
flake8 src/ tests/ scripts/ --max-line-length=100 --extend-ignore=E203,W503
```

### mypy

Type checking:

```bash
mypy src/ --ignore-missing-imports
```

## Git Commit Messages

Format:

```
<type>: <subject>

<body>

<footer>
```

**Types:**

- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation
- `style`: Formatting
- `refactor`: Code restructuring
- `test`: Tests
- `chore`: Maintenance

**Example:**

```
feat: add custom fitness function support

- Add FitnessCalculator base class
- Implement medical diagnosis fitness
- Update documentation

Closes #123
```
