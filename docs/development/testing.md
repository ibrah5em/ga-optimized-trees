# Testing Guide

Comprehensive testing practices.

## Test Structure

```
tests/
├── unit/                    # Unit tests
│   ├── test_genotype.py     # Tree structure tests
│   ├── test_ga_operators.py # GA operator tests
│   └── test_fitness.py      # Fitness calculation tests
├── integration/             # Integration tests
│   └── test_end_to_end.py   # Full workflow tests
└── conftest.py              # Shared fixtures
```

## Running Tests

```bash
# All tests
pytest tests/ -v

# Specific category
pytest tests/unit/ -v
pytest tests/integration/ -v

# Specific file
pytest tests/unit/test_genotype.py -v

# With coverage
pytest tests/ -v --cov=src/ga_trees --cov-report=html

# Parallel execution
pytest tests/ -n auto
```

## Writing Tests

### AAA Pattern

**Arrange-Act-Assert:**

```python
def test_tree_crossover():
    # Arrange
    parent1 = create_sample_tree(depth=2)
    parent2 = create_sample_tree(depth=2)

    # Act
    child1, child2 = Crossover.subtree_crossover(parent1, parent2)

    # Assert
    assert child1.get_depth() <= parent1.max_depth
    assert child2.get_depth() <= parent2.max_depth
    assert child1 is not parent1  # New instance
```

### Test Naming

```python
# Good - descriptive
def test_mutation_reduces_tree_size_when_pruning():
    pass


def test_fitness_increases_with_accuracy():
    pass


# Bad - vague
def test_mutation():
    pass


def test_tree():
    pass
```

### Fixtures

**Define in `conftest.py`:**

```python
import pytest
from sklearn.datasets import load_iris


@pytest.fixture
def iris_data():
    """Load Iris dataset."""
    return load_iris(return_X_y=True)


@pytest.fixture
def simple_tree():
    """Create simple 3-node tree."""
    left = create_leaf_node(0, 1)
    right = create_leaf_node(1, 1)
    root = create_internal_node(0, 0.5, left, right, 0)
    return TreeGenotype(root=root, n_features=4, n_classes=2)
```

**Use in tests:**

```python
def test_tree_prediction(simple_tree, iris_data):
    X, y = iris_data
    predictor = TreePredictor()
    y_pred = predictor.predict(simple_tree, X)
    assert len(y_pred) == len(y)
```

## Test Coverage

**Target:** 80%+ overall

**Check coverage:**

```bash
pytest tests/ --cov=src/ga_trees --cov-report=term-missing
```

**View HTML report:**

```bash
pytest tests/ --cov=src/ga_trees --cov-report=html
open htmlcov/index.html
```

## Best Practices

1. **Test public API**: Don't test private methods
1. **One assertion per concept**: Split complex tests
1. **Descriptive names**: Test name explains what's tested
1. **Fast tests**: Use small datasets, mock when needed
1. **Isolated tests**: No dependencies between tests
1. **Deterministic**: Set random seeds
