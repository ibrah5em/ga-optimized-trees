"""
Complete GA Engine Implementation with All Operators

This file contains the full genetic algorithm engine including:
- Population initialization
- Selection operators
- Crossover operators
- Mutation operators
- Main evolution loop
"""

import logging
import random
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from ga_trees.ga.improved_crossover import safe_subtree_crossover
from ga_trees.ga.split_points import (
    MIDPOINT_STRATEGY,
    UNIFORM_STRATEGY,
    candidate_thresholds,
    sample_threshold,
    samples_reaching,
    step_threshold,
    validate_split_strategy,
)
from ga_trees.genotype.tree_genotype import (
    Node,
    TreeGenotype,
    create_internal_node,
    create_leaf_node,
)

logger = logging.getLogger(__name__)

#: Per-node probability that :class:`TreeInitializer` stops growing and emits a
#: leaf, independent of the depth/sample stopping criteria. It controls how
#: bushy the *initial* population is: 0.0 grows every branch to its structural
#: limit, higher values bias the population toward stumps.
DEFAULT_GROWTH_STOP_PROB = 0.3


def _fitness_key(tree: TreeGenotype) -> float:
    """Comparison key that ranks unevaluated individuals last.

    A truthiness test (``t.fitness_ if t.fitness_ else -inf``) buckets a
    legitimate fitness of exactly 0.0 with the unevaluated ones, which drops
    those individuals out of elitism, tournaments and the generation
    statistics. Only ``None`` means "not evaluated yet".
    """
    return tree.fitness_ if tree.fitness_ is not None else -np.inf


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
                "threshold_perturbation": 0.4,
                "feature_replacement": 0.3,
                "prune_subtree": 0.2,
                "expand_leaf": 0.1,
            }


class TreeInitializer:
    """Initialize random decision trees.

    Args:
        n_features: Number of input features.
        n_classes: Number of target classes (classification only).
        max_depth: Maximum depth of a generated tree.
        min_samples_split: Minimum samples required to split a node.
        min_samples_leaf: Minimum samples required in each child of a split.
        task_type: ``"classification"`` or ``"regression"``.
        growth_stop_prob: Per-node probability of stopping growth early, in
            [0, 1). Defaults to :data:`DEFAULT_GROWTH_STOP_PROB`.
        split_strategy: Where thresholds come from — ``"midpoint"`` samples the
            observed midpoints of the data reaching the node, ``"uniform"`` is
            a uniform draw across the feature's range. See
            :mod:`ga_trees.ga.split_points`.
    """

    def __init__(
        self,
        n_features: int,
        n_classes: int,
        max_depth: int,
        min_samples_split: int,
        min_samples_leaf: int,
        task_type: str = "classification",
        growth_stop_prob: float = DEFAULT_GROWTH_STOP_PROB,
        split_strategy: str = MIDPOINT_STRATEGY,
    ):
        if not (0.0 <= growth_stop_prob < 1.0):
            raise ValueError(f"growth_stop_prob must be in [0, 1), got {growth_stop_prob}.")

        self.n_features = n_features
        self.n_classes = n_classes
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.task_type = task_type
        self.growth_stop_prob = growth_stop_prob
        self.split_strategy = validate_split_strategy(split_strategy)

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
            or random.random() < self.growth_stop_prob
        )

        if should_stop:
            # Create leaf
            prediction = self._calculate_prediction(y)
            return create_leaf_node(prediction, depth)

        # Create internal node. The feature is drawn uniformly under either split
        # strategy: the strategy changes where the *threshold* comes from and
        # nothing else, so comparing strategies isolates that one change.
        feature_idx = random.randint(0, self.n_features - 1)

        threshold = sample_threshold(
            X[:, feature_idx],
            min_samples_leaf=self.min_samples_leaf,
            strategy=self.split_strategy,
        )
        if threshold is None:
            # The feature is constant here, or no split of it leaves
            # min_samples_leaf on both sides.
            prediction = self._calculate_prediction(y)
            return create_leaf_node(prediction, depth)

        left_mask = X[:, feature_idx] <= threshold
        right_mask = ~left_mask

        # Under the midpoint strategy the candidate set is pre-filtered on
        # min_samples_leaf, so this only ever fires for the uniform strategy.
        if np.sum(left_mask) < self.min_samples_leaf or np.sum(right_mask) < self.min_samples_leaf:
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
            winner = max(tournament, key=_fitness_key)
            selected.append(winner.copy())
        return selected

    @staticmethod
    def elitism_selection(population: List[TreeGenotype], n_elite: int) -> List[TreeGenotype]:
        """Select top n individuals."""
        sorted_pop = sorted(population, key=_fitness_key, reverse=True)
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
    """Mutation operators.

    Args:
        n_features: Number of input features.
        feature_ranges: Per-feature ``(min, max)``, used by the ``uniform``
            strategy and as the fallback when no training matrix is supplied.
        X: Training design matrix. Supplying it lets the threshold operators
            draw from the values actually reaching the node being mutated
            instead of the feature's global range. Pass the
            GA-training split, never the validation split — a threshold chosen
            from data the fitness is scored on leaks it into the search.
        min_samples_leaf: Minimum samples a split must leave on each side.
        split_strategy: ``"midpoint"`` or ``"uniform"``. See
            :mod:`ga_trees.ga.split_points`.
    """

    def __init__(
        self,
        n_features: int,
        feature_ranges: Dict[int, Tuple[float, float]],
        X: Optional[np.ndarray] = None,
        min_samples_leaf: int = 1,
        split_strategy: str = MIDPOINT_STRATEGY,
    ):
        self.n_features = n_features
        self.feature_ranges = feature_ranges
        self.X = X
        self.min_samples_leaf = min_samples_leaf
        self.split_strategy = validate_split_strategy(split_strategy)

    def _local_candidates(self, tree: TreeGenotype, node: Node, feature_idx: int) -> np.ndarray:
        """Valid thresholds for *feature_idx* among the samples reaching *node*.

        Falls back to the feature's marginal distribution when the node is
        unreachable — crossover can graft a subtree behind a test that no sample
        satisfies — and to an empty set when there is no training matrix to
        consult.
        """
        if self.X is None or self.split_strategy != MIDPOINT_STRATEGY:
            return np.empty(0, dtype=float)

        indices = samples_reaching(tree.root, node, self.X)
        if indices is not None and indices.size > 0:
            candidates = candidate_thresholds(self.X[indices, feature_idx], self.min_samples_leaf)
            if candidates.size:
                return candidates

        return candidate_thresholds(self.X[:, feature_idx], self.min_samples_leaf)

    def _draw_threshold(self, tree: TreeGenotype, node: Node, feature_idx: int) -> float:
        """Draw a fresh threshold for *feature_idx* at *node*."""
        candidates = self._local_candidates(tree, node, feature_idx)
        if candidates.size:
            return float(candidates[random.randrange(candidates.size)])

        if self.X is not None and self.split_strategy == UNIFORM_STRATEGY:
            drawn = sample_threshold(self.X[:, feature_idx], strategy=UNIFORM_STRATEGY)
            if drawn is not None:
                return drawn

        if feature_idx in self.feature_ranges:
            min_val, max_val = self.feature_ranges[feature_idx]
            return random.uniform(min_val, max_val)
        return 0.0

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
        """Perturb threshold of random internal node.

        With training data available the perturbed value is snapped back onto an
        observed split point, so the step always changes the partition. Without
        it, this is the original Gaussian jitter clipped to the feature range.
        """
        internal_nodes = tree.get_internal_nodes()
        if not internal_nodes:
            return tree

        node = random.choice(internal_nodes)
        if node.feature_idx is None or node.threshold is None:
            return tree

        candidates = self._local_candidates(tree, node, node.feature_idx)
        if candidates.size:
            node.threshold = step_threshold(float(node.threshold), candidates)
            return tree

        if node.feature_idx in self.feature_ranges:
            min_val, max_val = self.feature_ranges[node.feature_idx]
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

        # The old threshold belonged to the old feature and is meaningless on
        # the new one, so it is redrawn rather than carried over.
        node.threshold = self._draw_threshold(tree, node, new_feature)

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
        feature_idx = random.randint(0, self.n_features - 1)

        # Drawn while the node is still a leaf: samples_reaching stops at the
        # target, so routing is unaffected, but the intent is clearer this way.
        threshold = self._draw_threshold(tree, node, feature_idx)

        # Convert to internal
        node.node_type = "internal"
        node.feature_idx = feature_idx
        node.threshold = threshold

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
        repair: Optional[Callable[[TreeGenotype], TreeGenotype]] = None,
    ):
        self.config = config
        self.initializer = initializer
        self.fitness_function = fitness_function
        self.mutation = mutation
        # Optional in-place repair applied to every offspring after variation
        # (see ga_trees.ga.repair). None keeps the original
        # behaviour, where the sample-count constraints hold only at init.
        self.repair = repair
        self.population: List[TreeGenotype] = []
        self.best_individual: Optional[TreeGenotype] = None
        self.history: Dict[str, List] = {"best_fitness": [], "avg_fitness": [], "diversity": []}

    def initialize_population(self, X: np.ndarray, y: np.ndarray):
        """Create initial random population."""
        self.population = []
        for _ in range(self.config.population_size):
            tree = self.initializer.create_random_tree(X, y)
            self.population.append(tree)

    def _score(
        self,
        tree: TreeGenotype,
        X: np.ndarray,
        y: np.ndarray,
        X_val: Optional[np.ndarray],
        y_val: Optional[np.ndarray],
    ) -> float:
        """Score one individual, holding out the validation split if there is one.

        The four-argument call is only made when a validation set exists, so
        fitness functions written against the original three-argument signature
        keep working unchanged.
        """
        if X_val is None or y_val is None:
            return self.fitness_function(tree, X, y)
        return self.fitness_function(tree, X, y, X_val, y_val)

    def evaluate_population(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ):
        """Evaluate fitness for entire population."""
        for individual in self.population:
            if individual.fitness_ is None:
                individual.fitness_ = self._score(individual, X, y, X_val, y_val)

    def evolve(
        self,
        X: np.ndarray,
        y: np.ndarray,
        verbose: bool = True,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> TreeGenotype:
        """
        Main evolution loop.

        Args:
            X: Training features — leaf predictions are fitted on these.
            y: Training labels.
            verbose: Print progress.
            X_val: Optional held-out features to score fitness on. Without
                them, fitness is resubstitution: leaves are
                fitted and scored on the same rows, so the search rewards
                memorisation and prefers whichever tree overfits hardest.
                Structure and thresholds are chosen from ``X``/``y`` alone, so
                the validation split stays genuinely unseen by the search.
            y_val: Optional held-out labels.

        Returns:
            Best individual found, selected by validation fitness when a
            validation set is supplied.
        """
        if (X_val is None) != (y_val is None):
            raise ValueError("X_val and y_val must be supplied together.")

        # --- LDD-9: reproducibility ---
        if self.config.random_state is not None:
            random.seed(self.config.random_state)
            np.random.seed(self.config.random_state)

        # Initialize
        self.initialize_population(X, y)
        self.evaluate_population(X, y, X_val, y_val)

        stagnation_counter = 0
        previous_best_fitness = -np.inf

        for generation in range(self.config.n_generations):
            # Track statistics
            fitnesses = [ind.fitness_ for ind in self.population if ind.fitness_ is not None]
            if fitnesses:
                best_fitness = max(fitnesses)
                avg_fitness = np.mean(fitnesses)
                self.history["best_fitness"].append(best_fitness)
                self.history["avg_fitness"].append(avg_fitness)

                # Update best individual
                best_ind = max(self.population, key=_fitness_key)
                if self.best_individual is None or _fitness_key(best_ind) > _fitness_key(
                    self.best_individual
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
                        "Gen %d: Best=%.4f, Avg=%.4f", generation, best_fitness, avg_fitness
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

                # Mutation
                if random.random() < self.config.mutation_prob:
                    child1 = self.mutation.mutate(child1, self.config.mutation_types)
                if random.random() < self.config.mutation_prob:
                    child2 = self.mutation.mutate(child2, self.config.mutation_types)

                if self.repair is not None:
                    child1 = self.repair(child1)
                    child2 = self.repair(child2)

                # Reset fitness (will be evaluated next iteration)
                child1.fitness_ = None
                child2.fitness_ = None

                next_population.append(child1)
                if len(next_population) < self.config.population_size:
                    next_population.append(child2)

            self.population = next_population

            # Evaluate new individuals
            self.evaluate_population(X, y, X_val, y_val)

        return self.best_individual

    def get_history(self) -> Dict[str, List]:
        """Get evolution history."""
        return self.history
