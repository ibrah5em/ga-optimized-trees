"""
Pytest configuration and shared fixtures.

This file provides common test fixtures used across all tests.
"""

import pytest
import numpy as np
from sklearn.datasets import load_iris, load_wine, load_breast_cancer

from ga_trees.genotype.tree_genotype import TreeGenotype, create_leaf_node, create_internal_node
from ga_trees.ga.engine import TreeInitializer


@pytest.fixture
def random_seed():
    """Set random seed for reproducibility."""
    np.random.seed(42)
    import random
    random.seed(42)
    return 42


@pytest.fixture
def iris_dataset():
    """Load Iris dataset."""
    X, y = load_iris(return_X_y=True)
    return X, y


@pytest.fixture
def wine_dataset():
    """Load Wine dataset."""
    X, y = load_wine(return_X_y=True)
    return X, y


@pytest.fixture
def breast_cancer_dataset():
    """Load Breast Cancer dataset."""
    X, y = load_breast_cancer(return_X_y=True)
    return X, y


@pytest.fixture
def simple_tree():
    """Create a simple 3-node tree for testing."""
    left = create_leaf_node(prediction=0, depth=1)
    right = create_leaf_node(prediction=1, depth=1)
    root = create_internal_node(
        feature_idx=0,
        threshold=0.5,
        left_child=left,
        right_child=right,
        depth=0
    )
    
    tree = TreeGenotype(
        root=root,
        n_features=4,
        n_classes=2,
        max_depth=5,
        min_samples_split=10,
        min_samples_leaf=5
    )
    
    return tree


@pytest.fixture
def tree_initializer():
    """Create tree initializer for tests."""
    return TreeInitializer(
        n_features=4,
        n_classes=2,
        max_depth=5,
        min_samples_split=10,
        min_samples_leaf=5
    )


@pytest.fixture
def feature_ranges():
    """Create dummy feature ranges."""
    return {
        0: (0.0, 1.0),
        1: (0.0, 1.0),
        2: (0.0, 1.0),
        3: (0.0, 1.0),
    }


# Configure pytest
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )