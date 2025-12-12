"""
Complete GA Engine Implementation with All Operators

This file contains the full genetic algorithm engine including:
- Population initialization
- Selection operators
- Crossover operators
- Mutation operators
- Main evolution loop
"""

import copy
import random
from collections import deque
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

# Assuming tree_genotype.py is available
from ga_trees.ga.improved_crossover import safe_subtree_crossover
from ga_trees.genotype.tree_genotype import (
    Node,
    TreeGenotype,
    create_internal_node,
    create_leaf_node,
)


@dataclass
class GAConfig:
    """Configuration for genetic algorithm."""

    population_size: int = 100
    n_generations: int = 50
    crossover_prob: float = 0.7
    mutation_prob: float = 0.2
    tournament_size: int = 3
    elitism_ratio: float = 0.1
    mutation_types: Dict[str, float] = None

    def __post_init__(self):
        if self.mutation_types is None:
            self.mutation_types = {
                "threshold_perturbation": 0.4,
                "feature_replacement": 0.3,
                "prune_subtree": 0.2,
                "expand_leaf": 0.1,
            }


class TreeInitializer:
    """Initialize random decision trees."""

    def __init__(
        self,
        n_features: int,
        n_classes: int,
        max_depth: int,
        min_samples_split: int,
        min_samples_leaf: int,
        task_type: str = "classification",
    ):
        self.n_features = n_features
        self.n_classes = n_classes
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.task_type = task_type

    def create_random_tree(self, X: np.ndarray, y: np.ndarray) -> TreeGenotype:
        """Create a random valid tree."""
        root = self._grow_tree(X, y, depth=0)
        return TreeGenotype(
            root=root,
            n_features=self.n_features,
            n_classes=self.n_classes,
            task_type=self.task_type,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
        )

    def _grow_tree(self, X: np.ndarray, y: np.ndarray, depth: int) -> Node:
        """Recursively grow tree using random decisions."""
        n_samples = len(X)

        # Stopping criteria
        should_stop = (
            depth >= self.max_depth
            or n_samples < self.min_samples_split
            or len(np.unique(y)) == 1
            or random.random() < 0.3  # Random early stopping
        )

        if should_stop:
            # Create leaf
            prediction = self._calculate_prediction(y)
            return create_leaf_node(prediction, depth)

        # Create internal node
        feature_idx = random.randint(0, self.n_features - 1)

        # Get threshold from data
        feature_values = X[:, feature_idx]
        unique_vals = np.unique(feature_values)
        if len(unique_vals) > 1:
            threshold = random.uniform(float(np.min(feature_values)), float(np.max(feature_values)))
        else:
            # All values same, create leaf
            prediction = self._calculate_prediction(y)
            return create_leaf_node(prediction, depth)

        # Split data
        left_mask = X[:, feature_idx] <= threshold
        right_mask = ~left_mask

        if np.sum(left_mask) < self.min_samples_leaf or np.sum(right_mask) < self.min_samples_leaf:
            # Split too small, create leaf
            prediction = self._calculate_prediction(y)
            return create_leaf_node(prediction, depth)

        # Recursively create children
        left_child = self._grow_tree(X[left_mask], y[left_mask], depth + 1)
        right_child = self._grow_tree(X[right_mask], y[right_mask], depth + 1)

        return create_internal_node(feature_idx, threshold, left_child, right_child, depth)

    def _calculate_prediction(self, y: np.ndarray) -> Any:
        """Calculate leaf prediction."""
        if self.task_type == "classification":
            # Most common class
            unique, counts = np.unique(y, return_counts=True)
            return int(unique[np.argmax(counts)])
        else:
            # Mean for regression
            return float(np.mean(y))


class Selection:
    """Selection operators for GA."""

    @staticmethod
    def tournament_selection(
        population: List[TreeGenotype], tournament_size: int, n_select: int
    ) -> List[TreeGenotype]:
        """Tournament selection."""
        selected = []
        for _ in range(n_select):
            tournament = random.sample(population, tournament_size)
            winner = max(tournament, key=lambda t: t.fitness_ if t.fitness_ else -np.inf)
            selected.append(winner.copy())
        return selected

    @staticmethod
    def elitism_selection(population: List[TreeGenotype], n_elite: int) -> List[TreeGenotype]:
        """Select top n individuals."""
        sorted_pop = sorted(
            population, key=lambda t: t.fitness_ if t.fitness_ else -np.inf, reverse=True
        )
        return [ind.copy() for ind in sorted_pop[:n_elite]]


class Crossover:
    """Crossover operators."""

    @staticmethod
    def subtree_crossover(
        parent1: TreeGenotype, parent2: TreeGenotype
    ) -> Tuple[TreeGenotype, TreeGenotype]:
        """
        Perform subtree-aware crossover using improved method.
        """
        return safe_subtree_crossover(parent1, parent2)

    @staticmethod
    def _copy_node_contents(src: Node, dst: Node):
        """Copy contents from src to dst node."""
        dst.node_type = src.node_type
        dst.feature_idx = src.feature_idx
        dst.threshold = src.threshold
        dst.operator = src.operator
        dst.prediction = (
            src.prediction
            if src.prediction is None
            else (
                src.prediction.copy() if isinstance(src.prediction, np.ndarray) else src.prediction
            )
        )
        dst.left_child = src.left_child.copy() if src.left_child else None
        dst.right_child = src.right_child.copy() if src.right_child else None

    @staticmethod
    def _repair_tree(tree: TreeGenotype) -> TreeGenotype:
        """Repair tree to satisfy constraints."""
        # Fix depths
        Crossover._fix_depths(tree.root, 0)

        # Prune if too deep
        if tree.get_depth() > tree.max_depth:
            tree = Crossover._prune_to_depth(tree, tree.max_depth)

        return tree

    @staticmethod
    def _fix_depths(node: Node, depth: int):
        """Recursively fix depth values."""
        if node is None:
            return
        node.depth = depth
        if node.left_child:
            Crossover._fix_depths(node.left_child, depth + 1)
        if node.right_child:
            Crossover._fix_depths(node.right_child, depth + 1)

    @staticmethod
    def _prune_to_depth(tree: TreeGenotype, max_depth: int) -> TreeGenotype:
        """Prune tree to maximum depth."""

        def prune_node(node: Node, depth: int) -> Node:
            if node is None:
                return None
            if depth >= max_depth:
                # Convert to leaf
                return create_leaf_node(node.prediction if node.is_leaf() else 0, depth)
            if node.is_leaf():
                return node
            node.left_child = prune_node(node.left_child, depth + 1)
            node.right_child = prune_node(node.right_child, depth + 1)
            return node

        tree.root = prune_node(tree.root, 0)
        return tree


class Mutation:
    """Mutation operators."""

    def __init__(self, n_features: int, feature_ranges: Dict[int, Tuple[float, float]]):
        self.n_features = n_features
        self.feature_ranges = feature_ranges

    def mutate(self, tree: TreeGenotype, mutation_types: Dict[str, float]) -> TreeGenotype:
        """Apply mutation to tree based on probabilities."""
        tree = tree.copy()

        # Choose mutation type
        mut_type = random.choices(
            list(mutation_types.keys()), weights=list(mutation_types.values()), k=1
        )[0]

        if mut_type == "threshold_perturbation":
            tree = self.threshold_perturbation(tree)
        elif mut_type == "feature_replacement":
            tree = self.feature_replacement(tree)
        elif mut_type == "prune_subtree":
            tree = self.prune_subtree(tree)
        elif mut_type == "expand_leaf":
            tree = self.expand_leaf(tree)

        return tree

    def threshold_perturbation(self, tree: TreeGenotype) -> TreeGenotype:
        """Perturb threshold of random internal node."""
        internal_nodes = tree.get_internal_nodes()
        if not internal_nodes:
            return tree

        node = random.choice(internal_nodes)
        if node.feature_idx in self.feature_ranges:
            min_val, max_val = self.feature_ranges[node.feature_idx]
            # Gaussian perturbation
            std = max((max_val - min_val) * 0.1, 1e-6)  # Minimum variance
            new_threshold = node.threshold + random.gauss(0, std)
            node.threshold = np.clip(new_threshold, min_val, max_val)

        return tree

    def feature_replacement(self, tree: TreeGenotype) -> TreeGenotype:
        """Replace feature in random internal node."""
        internal_nodes = tree.get_internal_nodes()
        if not internal_nodes:
            return tree

        node = random.choice(internal_nodes)
        new_feature = random.randint(0, self.n_features - 1)
        node.feature_idx = new_feature

        # Update threshold to valid range
        if new_feature in self.feature_ranges:
            min_val, max_val = self.feature_ranges[new_feature]
            node.threshold = random.uniform(min_val, max_val)

        return tree

    def prune_subtree(self, tree: TreeGenotype) -> TreeGenotype:
        """Convert random internal node to leaf."""
        internal_nodes = tree.get_internal_nodes()
        if not internal_nodes or len(internal_nodes) <= 1:
            return tree  # Don't prune root if it's the only internal node

        node = random.choice(internal_nodes)
        # Convert to leaf
        node.node_type = "leaf"
        node.prediction = 0  # Will be updated during fitness evaluation
        node.left_child = None
        node.right_child = None
        node.feature_idx = None
        node.threshold = None

        return tree

    def expand_leaf(self, tree: TreeGenotype) -> TreeGenotype:
        """Convert random leaf to internal node (if depth allows)."""
        leaves = tree.get_all_leaves()
        expandable_leaves = [l for l in leaves if l.depth < tree.max_depth - 1]

        if not expandable_leaves:
            return tree

        node = random.choice(expandable_leaves)
        # Convert to internal
        node.node_type = "internal"
        node.feature_idx = random.randint(0, self.n_features - 1)

        if node.feature_idx in self.feature_ranges:
            min_val, max_val = self.feature_ranges[node.feature_idx]
            node.threshold = random.uniform(min_val, max_val)
        else:
            node.threshold = 0.0

        # Create children
        node.left_child = create_leaf_node(node.prediction, node.depth + 1)
        node.right_child = create_leaf_node(node.prediction, node.depth + 1)
        node.prediction = None

        return tree


class GAEngine:
    """Main genetic algorithm engine."""

    def __init__(
        self,
        config: GAConfig,
        initializer: TreeInitializer,
        fitness_function: Callable[[TreeGenotype, np.ndarray, np.ndarray], float],
        mutation: Mutation,
    ):
        self.config = config
        self.initializer = initializer
        self.fitness_function = fitness_function
        self.mutation = mutation
        self.population: List[TreeGenotype] = []
        self.best_individual: Optional[TreeGenotype] = None
        self.history: Dict[str, List] = {"best_fitness": [], "avg_fitness": [], "diversity": []}

    def initialize_population(self, X: np.ndarray, y: np.ndarray):
        """Create initial random population."""
        self.population = []
        for _ in range(self.config.population_size):
            tree = self.initializer.create_random_tree(X, y)
            self.population.append(tree)

    def evaluate_population(self, X: np.ndarray, y: np.ndarray):
        """Evaluate fitness for entire population."""
        for individual in self.population:
            if individual.fitness_ is None:
                individual.fitness_ = self.fitness_function(individual, X, y)

    def evolve(self, X: np.ndarray, y: np.ndarray, verbose: bool = True) -> TreeGenotype:
        """
        Main evolution loop.

        Args:
            X: Training features
            y: Training labels
            verbose: Print progress

        Returns:
            Best individual found
        """
        # Initialize
        self.initialize_population(X, y)
        self.evaluate_population(X, y)

        for generation in range(self.config.n_generations):
            # Track statistics
            fitnesses = [ind.fitness_ for ind in self.population if ind.fitness_]
            if fitnesses:
                best_fitness = max(fitnesses)
                avg_fitness = np.mean(fitnesses)
                self.history["best_fitness"].append(best_fitness)
                self.history["avg_fitness"].append(avg_fitness)

                # Update best individual
                best_ind = max(self.population, key=lambda t: t.fitness_ if t.fitness_ else -np.inf)
                if (
                    self.best_individual is None
                    or best_ind.fitness_ > self.best_individual.fitness_
                ):
                    self.best_individual = best_ind.copy()

                if verbose and generation % 10 == 0:
                    print(f"Gen {generation}: Best={best_fitness:.4f}, Avg={avg_fitness:.4f}")

            # Create next generation
            next_population = []

            # Elitism
            n_elite = int(self.config.elitism_ratio * self.config.population_size)
            if n_elite > 0:
                elite = Selection.elitism_selection(self.population, n_elite)
                next_population.extend(elite)

            # Generate offspring
            while len(next_population) < self.config.population_size:
                # Selection
                parents = Selection.tournament_selection(
                    self.population, self.config.tournament_size, n_select=2
                )

                # Crossover
                if random.random() < self.config.crossover_prob:
                    child1, child2 = Crossover.subtree_crossover(parents[0], parents[1])
                else:
                    child1, child2 = parents[0].copy(), parents[1].copy()

                # Mutation
                if random.random() < self.config.mutation_prob:
                    child1 = self.mutation.mutate(child1, self.config.mutation_types)
                if random.random() < self.config.mutation_prob:
                    child2 = self.mutation.mutate(child2, self.config.mutation_types)

                # Reset fitness (will be evaluated next iteration)
                child1.fitness_ = None
                child2.fitness_ = None

                next_population.extend([child1, child2])

            # Trim to population size
            self.population = next_population[: self.config.population_size]

            # Evaluate new individuals
            self.evaluate_population(X, y)

        return self.best_individual

    def get_history(self) -> Dict[str, List]:
        """Get evolution history."""
        return self.history
