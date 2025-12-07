"""Unit tests for tree genotype."""

import numpy as np
import pytest

from ga_trees.genotype.tree_genotype import (
    Node,
    TreeGenotype,
    create_internal_node,
    create_leaf_node,
)


class TestNode:
    """Test Node class."""

    def test_create_leaf(self):
        """Test leaf node creation."""
        leaf = create_leaf_node(prediction=1, depth=0)
        assert leaf.is_leaf()
        assert not leaf.is_internal()
        assert leaf.prediction == 1
        assert leaf.depth == 0

    def test_create_internal(self):
        """Test internal node creation."""
        left = create_leaf_node(0, depth=1)
        right = create_leaf_node(1, depth=1)
        internal = create_internal_node(
            feature_idx=0, threshold=0.5, left_child=left, right_child=right, depth=0
        )

        assert internal.is_internal()
        assert not internal.is_leaf()
        assert internal.feature_idx == 0
        assert internal.threshold == 0.5
        assert internal.left_child == left
        assert internal.right_child == right

    def test_node_height(self):
        """Test height calculation."""
        # Single leaf
        leaf = create_leaf_node(0, 0)
        assert leaf.get_height() == 0

        # Tree with depth 2
        left = create_leaf_node(0, 2)
        right = create_leaf_node(1, 2)
        internal1 = create_internal_node(1, 0.5, left, right, 1)
        leaf2 = create_leaf_node(0, 2)
        root = create_internal_node(0, 0.3, internal1, leaf2, 0)

        assert root.get_height() == 2

    def test_count_nodes(self):
        """Test node counting."""
        leaf = create_leaf_node(0, 0)
        assert leaf.count_nodes() == 1

        left = create_leaf_node(0, 1)
        right = create_leaf_node(1, 1)
        root = create_internal_node(0, 0.5, left, right, 0)
        assert root.count_nodes() == 3


class TestTreeGenotype:
    """Test TreeGenotype class."""

    def test_create_simple_tree(self):
        """Test creating a simple tree."""
        left = create_leaf_node(0, 1)
        right = create_leaf_node(1, 1)
        root = create_internal_node(0, 0.5, left, right, 0)

        tree = TreeGenotype(
            root=root, n_features=4, n_classes=2, task_type="classification", max_depth=5
        )

        assert tree.get_depth() == 1
        assert tree.get_num_nodes() == 3
        assert tree.get_num_leaves() == 2

    def test_get_all_nodes(self):
        """Test getting all nodes."""
        left = create_leaf_node(0, 1)
        right = create_leaf_node(1, 1)
        root = create_internal_node(0, 0.5, left, right, 0)

        tree = TreeGenotype(root=root, n_features=4, n_classes=2)
        nodes = tree.get_all_nodes()

        assert len(nodes) == 3
        assert root in nodes
        assert left in nodes
        assert right in nodes

    def test_features_used(self):
        """Test feature tracking."""
        left = create_leaf_node(0, 2)
        right = create_leaf_node(1, 2)
        internal = create_internal_node(1, 0.5, left, right, 1)
        leaf = create_leaf_node(0, 1)
        root = create_internal_node(0, 0.3, internal, leaf, 0)

        tree = TreeGenotype(root=root, n_features=4, n_classes=2)
        features = tree.get_features_used()

        assert features == {0, 1}
        assert tree.get_num_features_used() == 2

    def test_tree_balance(self):
        """Test balance calculation."""
        # Perfectly balanced tree
        l1 = create_leaf_node(0, 1)
        l2 = create_leaf_node(1, 1)
        root = create_internal_node(0, 0.5, l1, l2, 0)
        tree = TreeGenotype(root=root, n_features=4, n_classes=2)

        balance = tree.get_tree_balance()
        assert 0.9 <= balance <= 1.0  # Should be near perfect

    def test_validation_depth(self):
        """Test depth validation."""
        # Create tree that violates depth
        nodes = create_leaf_node(0, 0)
        for i in range(10):
            nodes = create_internal_node(0, 0.5, nodes, create_leaf_node(0, i + 1), i)

        tree = TreeGenotype(root=nodes, n_features=4, n_classes=2, max_depth=5)
        valid, errors = tree.validate()

        assert not valid
        assert any("depth" in str(e).lower() for e in errors)

    def test_to_rules(self):
        """Test rule extraction."""
        left = create_leaf_node(0, 1)
        right = create_leaf_node(1, 1)
        root = create_internal_node(0, 0.5, left, right, 0)

        tree = TreeGenotype(root=root, n_features=4, n_classes=2)
        rules = tree.to_rules()

        assert len(rules) == 2
        assert all("IF" in rule and "THEN" in rule for rule in rules)
