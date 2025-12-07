"""Test GA operators."""

import numpy as np
import pytest

from ga_trees.ga.engine import Crossover, Mutation, Selection, TreeInitializer
from ga_trees.genotype.tree_genotype import TreeGenotype, create_internal_node, create_leaf_node


class TestSelection:
    """Test selection operators."""

    def create_population(self, n=10):
        """Create dummy population."""
        pop = []
        for i in range(n):
            left = create_leaf_node(0, 1)
            right = create_leaf_node(1, 1)
            root = create_internal_node(0, 0.5, left, right, 0)
            tree = TreeGenotype(root=root, n_features=4, n_classes=2)
            tree.fitness_ = float(i) / n  # Increasing fitness
            pop.append(tree)
        return pop

    def test_tournament_selection(self):
        """Test tournament selection."""
        pop = self.create_population(10)
        selected = Selection.tournament_selection(pop, tournament_size=3, n_select=5)

        assert len(selected) == 5
        # Higher fitness should be more likely selected
        avg_fitness = np.mean([t.fitness_ for t in selected])
        assert avg_fitness > 0.3  # Should be above random average

    def test_elitism_selection(self):
        """Test elitism selection."""
        pop = self.create_population(10)
        elite = Selection.elitism_selection(pop, n_elite=3)

        assert len(elite) == 3
        # Should get top 3
        fitnesses = [t.fitness_ for t in elite]
        assert fitnesses == sorted(fitnesses, reverse=True)
        assert fitnesses[0] >= 0.9  # Best individual


class TestCrossover:
    """Test crossover operators."""

    def create_tree(self, depth=2):
        """Create a tree of given depth."""
        if depth == 0:
            return create_leaf_node(0, 0)
        left = self.create_tree(depth - 1)
        right = self.create_tree(depth - 1)
        return create_internal_node(0, 0.5, left, right, 0)

    def test_subtree_crossover(self):
        """Test subtree crossover."""
        root1 = self.create_tree(2)
        root2 = self.create_tree(2)

        tree1 = TreeGenotype(root=root1, n_features=4, n_classes=2, max_depth=5)
        tree2 = TreeGenotype(root=root2, n_features=4, n_classes=2, max_depth=5)

        child1, child2 = Crossover.subtree_crossover(tree1, tree2)

        assert child1 is not tree1
        assert child2 is not tree2
        assert child1.get_depth() <= child1.max_depth
        assert child2.get_depth() <= child2.max_depth


class TestMutation:
    """Test mutation operators."""

    def test_threshold_perturbation(self):
        """Test threshold mutation."""
        left = create_leaf_node(0, 1)
        right = create_leaf_node(1, 1)
        root = create_internal_node(0, 0.5, left, right, 0)
        tree = TreeGenotype(root=root, n_features=4, n_classes=2)

        feature_ranges = {i: (0.0, 1.0) for i in range(4)}
        mutation = Mutation(n_features=4, feature_ranges=feature_ranges)

        original_threshold = tree.root.threshold
        mutated = mutation.threshold_perturbation(tree)

        assert mutated.root.threshold != original_threshold
        assert 0.0 <= mutated.root.threshold <= 1.0

    def test_prune_subtree(self):
        """Test pruning mutation."""
        # Create tree with internal nodes
        l1 = create_leaf_node(0, 2)
        l2 = create_leaf_node(1, 2)
        internal = create_internal_node(1, 0.5, l1, l2, 1)
        leaf = create_leaf_node(0, 1)
        root = create_internal_node(0, 0.3, internal, leaf, 0)

        tree = TreeGenotype(root=root, n_features=4, n_classes=2, max_depth=5)
        original_nodes = tree.get_num_nodes()

        feature_ranges = {i: (0.0, 1.0) for i in range(4)}
        mutation = Mutation(n_features=4, feature_ranges=feature_ranges)

        mutated = mutation.prune_subtree(tree)

        # Tree should be smaller or same size
        assert mutated.get_num_nodes() <= original_nodes


class TestTreeInitializer:
    """Test tree initialization."""

    def test_create_random_tree(self):
        """Test random tree creation."""
        X = np.random.rand(100, 4)
        y = np.random.randint(0, 2, 100)

        initializer = TreeInitializer(
            n_features=4, n_classes=2, max_depth=5, min_samples_split=10, min_samples_leaf=5
        )

        tree = initializer.create_random_tree(X, y)

        assert tree.get_depth() <= 5
        assert tree.get_num_nodes() >= 1
        valid, errors = tree.validate()
        assert valid, f"Tree validation failed: {errors}"
