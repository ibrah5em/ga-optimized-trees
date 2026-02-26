"""
Complete GA Engine Implementation with All Operators

This file contains the full genetic algorithm engine including:
- Population initialization
- Selection operators
- Crossover operators
- Mutation operators
- Main evolution loop
"""

import inspect
import logging
import random
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from sklearn.model_selection import train_test_split

from ga_trees.ga.improved_crossover import safe_subtree_crossover
from ga_trees.genotype.tree_genotype import (
    Node,
    TreeGenotype,
    create_internal_node,
    create_leaf_node,
)

logger = logging.getLogger(__name__)


@dataclass
class GAConfig:
    """Configuration for genetic algorithm.

    Attributes:
        population_size: Number of individuals per generation (must be > 0).
        n_generations: Number of evolutionary generations (must be > 0).
        crossover_prob: Probability of crossover, in [0, 1].
        mutation_prob: Probability of mutation, in [0, 1].
        tournament_size: Tournament selection size (must be >= 2).
        elitism_ratio: Fraction of population preserved as elite, in [0, 1).
        mutation_types: Mapping of mutation operator names to selection weights.
        random_state: Optional seed for reproducibility (LDD-9).
    """

    population_size: int = 100
    n_generations: int = 50
    crossover_prob: float = 0.7
    mutation_prob: float = 0.2
    tournament_size: int = 3
    elitism_ratio: float = 0.1
    mutation_types: Dict[str, float] = None
    random_state: Optional[int] = None
    early_stopping_rounds: Optional[int] = None
    early_stopping_tol: float = 1e-6

    def __post_init__(self):
        # --- LDD-5: input validation ---
        if self.population_size <= 0:
            raise ValueError(f"population_size must be > 0, got {self.population_size}.")
        if self.n_generations <= 0:
            raise ValueError(f"n_generations must be > 0, got {self.n_generations}.")
        if not (0.0 <= self.crossover_prob <= 1.0):
            raise ValueError(f"crossover_prob must be in [0, 1], got {self.crossover_prob}.")
        if not (0.0 <= self.mutation_prob <= 1.0):
            raise ValueError(f"mutation_prob must be in [0, 1], got {self.mutation_prob}.")
        if self.tournament_size < 2:
            raise ValueError(f"tournament_size must be >= 2, got {self.tournament_size}.")
        if not (0.0 <= self.elitism_ratio < 1.0):
            raise ValueError(f"elitism_ratio must be in [0, 1), got {self.elitism_ratio}.")

        if self.mutation_types is None:
            self.mutation_types = {
                "threshold_perturbation": 0.30,
                "feature_replacement": 0.20,
                "prune_subtree": 0.15,
                "expand_leaf": 0.10,
                "subtree_regeneration": 0.10,
                "swap_children": 0.05,
                "hoist": 0.05,
                "smart_threshold": 0.05,
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

    def create_random_tree(self, X: np.ndarray, y: np.ndarray, mode: str = "grow") -> TreeGenotype:
        """Create a random valid tree.

        Args:
            X: Feature matrix.
            y: Labels.
            mode: ``"grow"`` uses depth-dependent early stopping (§1.1);
                ``"full"`` always expands to max_depth (ramped half-and-half).
        """
        root = self._grow_tree(X, y, depth=0, mode=mode)
        return TreeGenotype(
            root=root,
            n_features=self.n_features,
            n_classes=self.n_classes,
            task_type=self.task_type,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
        )

    def _grow_tree(self, X: np.ndarray, y: np.ndarray, depth: int, mode: str = "grow") -> Node:
        """Recursively grow tree using random decisions.

        Args:
            X: Feature matrix for this node's samples.
            y: Labels for this node's samples.
            depth: Current depth.
            mode: ``"grow"`` applies depth-dependent stopping probability;
                ``"full"`` skips random early stopping.
        """
        n_samples = len(X)
        # §1.1: depth-dependent stopping instead of fixed 30%
        early_stop = mode == "grow" and random.random() < (depth / max(self.max_depth, 1)) ** 2

        # Stopping criteria
        should_stop = (
            depth >= self.max_depth
            or n_samples < self.min_samples_split
            or len(np.unique(y)) == 1
            or early_stop
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
        left_child = self._grow_tree(X[left_mask], y[left_mask], depth + 1, mode=mode)
        right_child = self._grow_tree(X[right_mask], y[right_mask], depth + 1, mode=mode)

        return create_internal_node(feature_idx, threshold, left_child, right_child, depth)

    # ------------------------------------------------------------------
    # §1.2: Data-informed (seeded) initialization
    # ------------------------------------------------------------------

    def create_informed_tree(self, X: np.ndarray, y: np.ndarray) -> TreeGenotype:
        """Create a tree with information-gain-based splits (§1.2).

        For each internal node, with probability 0.5 picks the best
        feature/threshold by Gini impurity, and with probability 0.5
        uses a random feature with a data-informed quantile threshold.
        """
        root = self._grow_informed_tree(X, y, depth=0)
        return TreeGenotype(
            root=root,
            n_features=self.n_features,
            n_classes=self.n_classes,
            task_type=self.task_type,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
        )

    def _grow_informed_tree(self, X: np.ndarray, y: np.ndarray, depth: int) -> Node:
        """Recursively grow a tree using Gini-informed splits."""
        n_samples = len(X)

        should_stop = (
            depth >= self.max_depth
            or n_samples < self.min_samples_split
            or len(np.unique(y)) == 1
            or random.random() < (depth / max(self.max_depth, 1)) ** 2
        )

        if should_stop:
            return create_leaf_node(self._calculate_prediction(y), depth)

        if random.random() < 0.5:
            # Use best Gini split
            feature_idx, threshold = self._best_gini_split(X, y)
        else:
            # Random feature, data-informed quantile threshold
            feature_idx = random.randint(0, self.n_features - 1)
            feature_values = X[:, feature_idx]
            unique_vals = np.unique(feature_values)
            if len(unique_vals) <= 1:
                return create_leaf_node(self._calculate_prediction(y), depth)
            q = random.uniform(0.1, 0.9)
            threshold = float(np.quantile(feature_values, q))

        if feature_idx is None or threshold is None:
            return create_leaf_node(self._calculate_prediction(y), depth)

        left_mask = X[:, feature_idx] <= threshold
        right_mask = ~left_mask

        if np.sum(left_mask) < self.min_samples_leaf or np.sum(right_mask) < self.min_samples_leaf:
            return create_leaf_node(self._calculate_prediction(y), depth)

        left_child = self._grow_informed_tree(X[left_mask], y[left_mask], depth + 1)
        right_child = self._grow_informed_tree(X[right_mask], y[right_mask], depth + 1)
        return create_internal_node(feature_idx, threshold, left_child, right_child, depth)

    def _best_gini_split(
        self, X: np.ndarray, y: np.ndarray
    ) -> Tuple[Optional[int], Optional[float]]:
        """Find best feature/threshold by Gini impurity reduction.

        Tries a random subset of features (sqrt selection) with a few
        candidate thresholds each, returning ``(feature_idx, threshold)``.
        """
        best_gini = np.inf
        best_feature: Optional[int] = None
        best_threshold: Optional[float] = None

        n_features_to_try = max(1, int(np.sqrt(self.n_features)))
        features_to_try = random.sample(range(self.n_features), n_features_to_try)

        for feature_idx in features_to_try:
            feature_values = X[:, feature_idx]
            unique_vals = np.unique(feature_values)
            if len(unique_vals) <= 1:
                continue

            if len(unique_vals) <= 10:
                thresholds = [
                    (unique_vals[i] + unique_vals[i + 1]) / 2.0 for i in range(len(unique_vals) - 1)
                ]
            else:
                thresholds = [
                    float(np.quantile(feature_values, q)) for q in [0.2, 0.4, 0.5, 0.6, 0.8]
                ]

            for threshold in thresholds:
                left_mask = feature_values <= threshold
                right_mask = ~left_mask
                n_left = int(np.sum(left_mask))
                n_right = int(np.sum(right_mask))

                if n_left < self.min_samples_leaf or n_right < self.min_samples_leaf:
                    continue

                n_total = len(y)
                gini = self._gini_impurity(y[left_mask], n_left, n_total) + self._gini_impurity(
                    y[right_mask], n_right, n_total
                )

                if gini < best_gini:
                    best_gini = gini
                    best_feature = feature_idx
                    best_threshold = threshold

        return best_feature, best_threshold

    def _gini_impurity(self, y: np.ndarray, n_subset: int, n_total: int) -> float:
        """Compute weighted Gini impurity for a node subset."""
        if n_subset == 0:
            return 0.0
        _, counts = np.unique(y, return_counts=True)
        probs = counts / n_subset
        return float((1.0 - np.sum(probs**2)) * (n_subset / n_total))

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
                # Convert to leaf — preserve prediction if leaf, else use None
                # (will be corrected by fit_leaf_predictions during evaluation)
                pred = node.prediction if node.is_leaf() else None
                leaf = create_leaf_node(pred if pred is not None else 0, depth)
                return leaf
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
        elif mut_type == "subtree_regeneration":
            tree = self.subtree_regeneration(tree)
        elif mut_type == "swap_children":
            tree = self.swap_children(tree)
        elif mut_type == "hoist":
            tree = self.hoist_mutation(tree)
        elif mut_type == "smart_threshold":
            tree = self.smart_threshold(tree)

        return tree

    def threshold_perturbation(self, tree: TreeGenotype) -> TreeGenotype:
        """Perturb threshold of random internal node.

        §3.1: Uses adaptive perturbation magnitude ``random.uniform(0.01, 0.2)``
        of the feature range instead of a fixed 10%.
        """
        internal_nodes = tree.get_internal_nodes()
        if not internal_nodes:
            return tree

        node = random.choice(internal_nodes)
        if node.feature_idx in self.feature_ranges:
            min_val, max_val = self.feature_ranges[node.feature_idx]
            # §3.1: adaptive perturbation magnitude
            scale = random.uniform(0.01, 0.2)
            std = max((max_val - min_val) * scale, 1e-6)
            new_threshold = node.threshold + random.gauss(0, std)
            node.threshold = float(np.clip(new_threshold, min_val, max_val))

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
        """Convert random internal node to leaf.

        LDD-13: Root node is excluded from candidates to prevent
        accidentally converting the entire tree into a single leaf.
        """
        internal_nodes = tree.get_internal_nodes()
        # LDD-13: exclude root from candidates
        candidates = [n for n in internal_nodes if n.node_id != tree.root.node_id]
        if not candidates:
            return tree  # Nothing to prune (root-only or single internal node)

        node = random.choice(candidates)
        # Inherit prediction from the leftmost leaf descendant
        descendant = node
        while descendant and not descendant.is_leaf():
            descendant = descendant.left_child
        inherited_pred = (
            descendant.prediction if (descendant and descendant.prediction is not None) else 0
        )
        # Convert to leaf
        node.node_type = "leaf"
        node.prediction = inherited_pred
        node.left_child = None
        node.right_child = None
        node.feature_idx = None
        node.threshold = None

        return tree

    def expand_leaf(self, tree: TreeGenotype) -> TreeGenotype:
        """Convert random leaf to internal node (if depth allows)."""
        leaves = tree.get_all_leaves()
        expandable_leaves = [leaf for leaf in leaves if leaf.depth < tree.max_depth - 1]

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

    # ------------------------------------------------------------------
    # §3.2: New mutation operators
    # ------------------------------------------------------------------

    def subtree_regeneration(self, tree: TreeGenotype) -> TreeGenotype:
        """Replace a random non-root subtree with a freshly generated one (§3.2).

        The new subtree is built using ``feature_ranges`` alone (no data),
        with depth-dependent stopping analogous to ``_grow_tree`` "grow" mode.
        """
        internal_nodes = tree.get_internal_nodes()
        candidates = [n for n in internal_nodes if n.node_id != tree.root.node_id]
        if not candidates:
            return tree

        node = random.choice(candidates)
        if node.depth >= tree.max_depth:
            return tree

        new_sub = self._generate_random_subtree(node.depth, tree.max_depth)
        node.node_type = new_sub.node_type
        node.feature_idx = new_sub.feature_idx
        node.threshold = new_sub.threshold
        node.prediction = new_sub.prediction
        node.left_child = new_sub.left_child
        node.right_child = new_sub.right_child
        return tree

    def _generate_random_subtree(self, current_depth: int, max_depth: int) -> Node:
        """Generate a random subtree using feature_ranges (no data)."""
        stop_prob = (current_depth / max(max_depth, 1)) ** 2
        if current_depth >= max_depth or random.random() < stop_prob:
            return create_leaf_node(0, current_depth)

        feature_idx = random.randint(0, self.n_features - 1)
        if feature_idx in self.feature_ranges:
            min_val, max_val = self.feature_ranges[feature_idx]
            threshold = random.uniform(min_val, max_val)
        else:
            threshold = 0.0

        left = self._generate_random_subtree(current_depth + 1, max_depth)
        right = self._generate_random_subtree(current_depth + 1, max_depth)
        return create_internal_node(feature_idx, threshold, left, right, current_depth)

    def swap_children(self, tree: TreeGenotype) -> TreeGenotype:
        """Swap left and right children of a random internal node (§3.2)."""
        internal_nodes = tree.get_internal_nodes()
        if not internal_nodes:
            return tree

        node = random.choice(internal_nodes)
        node.left_child, node.right_child = node.right_child, node.left_child
        return tree

    def hoist_mutation(self, tree: TreeGenotype) -> TreeGenotype:
        """Replace the tree's root with a random non-root subtree (§3.2).

        This is a natural tree-simplification operator — hoisting a subtree
        that already achieves good splits at a deeper level.
        """
        if tree.root.is_leaf():
            return tree

        all_nodes = tree.get_all_nodes()
        non_root = [n for n in all_nodes if n.node_id != tree.root.node_id]
        if not non_root:
            return tree

        replacement = random.choice(non_root).copy()
        tree.root = replacement
        Crossover._fix_depths(tree.root, 0)
        tree._assign_node_ids(tree.root, 0)
        return tree

    def smart_threshold(self, tree: TreeGenotype) -> TreeGenotype:
        """Set threshold to a data-meaningful quantile position (§3.1).

        Instead of Gaussian noise, picks from the 10th, 25th, 50th, 75th,
        or 90th percentile position within the feature's known range.
        """
        internal_nodes = tree.get_internal_nodes()
        if not internal_nodes:
            return tree

        node = random.choice(internal_nodes)
        if node.feature_idx in self.feature_ranges:
            min_val, max_val = self.feature_ranges[node.feature_idx]
            quantile_positions = [0.1, 0.25, 0.5, 0.75, 0.9]
            q = random.choice(quantile_positions)
            node.threshold = min_val + q * (max_val - min_val)
        return tree


class GAEngine:
    """Main genetic algorithm engine."""

    def __init__(
        self,
        config: GAConfig,
        initializer: TreeInitializer,
        fitness_function: Callable,
        mutation: Mutation,
    ):
        self.config = config
        self.initializer = initializer
        self.fitness_function = fitness_function
        self.mutation = mutation
        self.population: List[TreeGenotype] = []
        self.best_individual: Optional[TreeGenotype] = None
        self.history: Dict[str, List] = {"best_fitness": [], "avg_fitness": [], "diversity": []}
        # §1.3: detect whether fitness_function accepts X_val/y_val
        self._fitness_accepts_val: bool = self._check_fitness_accepts_val()

    def _check_fitness_accepts_val(self) -> bool:
        """Return True if ``fitness_function`` accepts validation data arguments.

        Uses ``inspect.signature`` to count parameters — a function that
        accepts at least 5 parameters (tree, X, y, X_val, y_val) will
        have validation data forwarded to it (§1.3).
        """
        try:
            sig = inspect.signature(self.fitness_function)
            return len(sig.parameters) >= 5
        except (ValueError, TypeError):
            return False

    def initialize_population(self, X: np.ndarray, y: np.ndarray):
        """Create initial population using ramped half-and-half with seeded trees (§1.1, §1.2).

        - 25% informed trees (Gini-based splits)
        - 37.5% "grow" mode random trees (depth-dependent stopping)
        - 37.5% "full" mode random trees (always expand to max_depth)
        """
        self.population = []
        n_informed = int(0.25 * self.config.population_size)
        n_random = self.config.population_size - n_informed
        n_grow = n_random // 2
        n_full = n_random - n_grow

        # §1.2: Informed trees seed the population with reasonable solutions
        for _ in range(n_informed):
            try:
                tree = self.initializer.create_informed_tree(X, y)
            except Exception:
                tree = self.initializer.create_random_tree(X, y, mode="grow")
            self.population.append(tree)

        # §1.1: Grow mode — depth-dependent early stopping
        for _ in range(n_grow):
            self.population.append(self.initializer.create_random_tree(X, y, mode="grow"))

        # §1.1: Full mode — always expand to max_depth (ramped half-and-half)
        for _ in range(n_full):
            self.population.append(self.initializer.create_random_tree(X, y, mode="full"))

    def evaluate_population(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ):
        """Evaluate fitness for entire population.

        Args:
            X: Training features (used for leaf fitting).
            y: Training labels.
            X_val: Optional validation features (LDD-3 / §1.3).
            y_val: Optional validation labels.
        """
        for individual in self.population:
            if individual.fitness_ is None:
                if self._fitness_accepts_val and X_val is not None and y_val is not None:
                    individual.fitness_ = self.fitness_function(individual, X, y, X_val, y_val)
                else:
                    individual.fitness_ = self.fitness_function(individual, X, y)

    def _split_data(
        self, X: np.ndarray, y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
        """Split into train/val sets for generalization-aware evaluation (§1.3).

        Returns ``(X_train, y_train, X_val, y_val)``.  Returns ``None`` val
        arrays when the dataset is too small to split meaningfully.
        """
        if len(X) < 20:
            return X, y, None, None
        try:
            unique_classes = np.unique(y)
            use_stratify = (
                y.ndim == 1
                and len(unique_classes) >= 2
                and all(int(np.sum(y == c)) >= 2 for c in unique_classes)
            )
            stratify = y if use_stratify else None
            X_train, X_val, y_train, y_val = train_test_split(
                X,
                y,
                test_size=0.2,
                random_state=self.config.random_state,
                stratify=stratify,
            )
            return X_train, y_train, X_val, y_val
        except Exception:
            return X, y, None, None

    def _get_adaptive_params(self, stagnation_counter: int) -> Tuple[float, Dict[str, float]]:
        """Return effective mutation probability and type weights (§1.5).

        When stagnation exceeds 5 rounds, temporarily boosts mutation rate
        and exploration operators to escape local optima.
        """
        effective_prob = self.config.mutation_prob
        effective_types = dict(self.config.mutation_types)

        if stagnation_counter > 5:
            effective_prob = min(0.5, self.config.mutation_prob * 1.5)
            for key in ["expand_leaf", "feature_replacement", "subtree_regeneration"]:
                if key in effective_types:
                    effective_types[key] = effective_types[key] * 1.5
            total = sum(effective_types.values())
            if total > 0:
                effective_types = {k: v / total for k, v in effective_types.items()}

        return effective_prob, effective_types

    def _inject_immigrants(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray],
        y_val: Optional[np.ndarray],
    ):
        """Replace the weakest individuals with fresh random trees (§1.4).

        Restores population diversity when fitness standard deviation drops
        below the diversity threshold.
        """
        n_immigrants = max(1, int(0.10 * self.config.population_size))
        sorted_idx = sorted(
            range(len(self.population)),
            key=lambda i: self.population[i].fitness_
            if self.population[i].fitness_ is not None
            else -np.inf,
        )
        for i in range(min(n_immigrants, len(sorted_idx))):
            new_tree = self.initializer.create_random_tree(X_train, y_train, mode="grow")
            new_tree.fitness_ = None
            self.population[sorted_idx[i]] = new_tree
        self.evaluate_population(X_train, y_train, X_val, y_val)

    def evolve(self, X: np.ndarray, y: np.ndarray, verbose: bool = True) -> TreeGenotype:
        """Main evolution loop.

        Args:
            X: Training features.
            y: Training labels.
            verbose: Print progress to logger.

        Returns:
            Best individual found.
        """
        # --- LDD-9: reproducibility ---
        if self.config.random_state is not None:
            random.seed(self.config.random_state)
            np.random.seed(self.config.random_state)

        # §1.3: Split into train/validation for generalization-aware evaluation
        X_train, y_train, X_val, y_val = self._split_data(X, y)

        # Initialize
        self.initialize_population(X_train, y_train)
        self.evaluate_population(X_train, y_train, X_val, y_val)

        stagnation_counter = 0
        previous_best_fitness = -np.inf

        for generation in range(self.config.n_generations):
            # Track statistics
            fitnesses = [ind.fitness_ for ind in self.population if ind.fitness_ is not None]
            if fitnesses:
                best_fitness = max(fitnesses)
                avg_fitness = float(np.mean(fitnesses))
                # §1.4: track diversity as std of fitness values
                diversity = float(np.std(fitnesses)) if len(fitnesses) > 1 else 0.0

                self.history["best_fitness"].append(best_fitness)
                self.history["avg_fitness"].append(avg_fitness)
                self.history["diversity"].append(diversity)

                # Update best individual
                best_ind = max(
                    self.population,
                    key=lambda t: t.fitness_ if t.fitness_ is not None else -np.inf,
                )
                if (
                    self.best_individual is None
                    or best_ind.fitness_ > self.best_individual.fitness_
                ):
                    self.best_individual = best_ind.copy()

                # Early stopping check
                if self.config.early_stopping_rounds is not None:
                    if best_fitness - previous_best_fitness > self.config.early_stopping_tol:
                        stagnation_counter = 0
                    else:
                        stagnation_counter += 1

                    if stagnation_counter >= self.config.early_stopping_rounds:
                        if verbose:
                            logger.info(
                                "Early stopping at generation %d (no improvement for %d rounds)",
                                generation,
                                self.config.early_stopping_rounds,
                            )
                        break

                    previous_best_fitness = best_fitness

                if verbose and generation % 10 == 0:
                    logger.info(
                        "Gen %d: Best=%.4f, Avg=%.4f, Div=%.4f",
                        generation,
                        best_fitness,
                        avg_fitness,
                        diversity,
                    )

                # §1.4: Immigration when diversity is too low
                if diversity < 0.01:
                    self._inject_immigrants(X_train, y_train, X_val, y_val)

            # §1.5: Adaptive mutation parameters
            effective_mutation_prob, effective_mutation_types = self._get_adaptive_params(
                stagnation_counter
            )

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

                # §1.5: Mutation with adaptive rate/types
                if random.random() < effective_mutation_prob:
                    child1 = self.mutation.mutate(child1, effective_mutation_types)
                if random.random() < effective_mutation_prob:
                    child2 = self.mutation.mutate(child2, effective_mutation_types)

                # Reset fitness (will be evaluated next iteration)
                child1.fitness_ = None
                child2.fitness_ = None

                next_population.append(child1)
                if len(next_population) < self.config.population_size:
                    next_population.append(child2)

            self.population = next_population

            # Evaluate new individuals
            self.evaluate_population(X_train, y_train, X_val, y_val)

            # §4.2: Notify fitness_function of current evolution phase (curriculum)
            phase = (generation + 1) / self.config.n_generations
            fn = self.fitness_function
            if hasattr(fn, "__self__") and hasattr(fn.__self__, "set_evolution_phase"):
                fn.__self__.set_evolution_phase(phase)

        return self.best_individual

    def get_history(self) -> Dict[str, List]:
        """Get evolution history."""
        return self.history
