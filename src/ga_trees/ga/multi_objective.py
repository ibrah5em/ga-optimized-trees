"""Multi-objective optimization with NSGA-II.

This implements Pareto-based optimization for the accuracy vs
interpretability trade-off using DEAP's NSGA-II routines.

Changes from original:
- LDD-2:   Crossover and mutation are now functional (were ``pass`` stubs).
- LDD-9:   Random seed support for reproducibility.
- LDD-14:  DEAP ``creator`` global state is cleaned up before re-creation.
- LDD-15:  ``toolbox.clone`` is properly registered.
- LDD-16:  Crowding distance is assigned before first ``selTournamentDCD``.
"""

import copy
import random
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from deap import base, creator, tools

from ga_trees.ga.improved_crossover import safe_subtree_crossover
from ga_trees.genotype.tree_genotype import TreeGenotype


def _cleanup_deap_creator() -> None:
    """Remove previously created DEAP creator classes to avoid
    stale global state (LDD-14)."""
    for name in ("Individual", "FitnessMulti"):
        if hasattr(creator, name):
            delattr(creator, name)


class ParetoOptimizer:
    """NSGA-II optimizer for multi-objective tree evolution.

    Evolves a population of decision trees simultaneously optimising
    accuracy **and** interpretability.  Returns a Pareto front of
    non-dominated solutions.

    Args:
        initializer: A ``TreeInitializer`` instance used to create
            the initial random population.
        fitness_fn: A callable ``(tree, X, y) -> (accuracy, interpretability)``
            that returns **both** objectives as a tuple.
        mutation_fn: A callable ``(tree) -> tree`` that returns a mutated copy.
        crossover_prob: Probability of applying crossover to a pair of offspring.
        mutation_prob: Probability of applying mutation to an individual.
        random_state: Optional seed for reproducibility (LDD-9).
    """

    def __init__(
        self,
        initializer,
        fitness_fn: Callable[..., Tuple[float, float]],
        mutation_fn: Callable[[TreeGenotype], TreeGenotype],
        crossover_prob: float = 0.7,
        mutation_prob: float = 0.2,
        random_state: Optional[int] = None,
    ):
        self.initializer = initializer
        self.fitness_fn = fitness_fn
        self.mutation_fn = mutation_fn
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.random_state = random_state

        # --- LDD-14: clean global state before creating ---
        _cleanup_deap_creator()
        # Maximise both accuracy and interpretability
        creator.create("FitnessMulti", base.Fitness, weights=(1.0, 1.0))
        creator.create("Individual", list, fitness=creator.FitnessMulti)

        self.toolbox = base.Toolbox()
        # --- LDD-15: register clone ---
        self.toolbox.register("clone", copy.deepcopy)

    # ------------------------------------------------------------------
    # Core NSGA-II loop
    # ------------------------------------------------------------------

    def evolve_pareto_front(
        self,
        X: np.ndarray,
        y: np.ndarray,
        population_size: int = 100,
        n_generations: int = 50,
        verbose: bool = False,
    ) -> List[TreeGenotype]:
        """Evolve Pareto-optimal solutions.

        Args:
            X: Training features.
            y: Training labels.
            population_size: Number of individuals per generation.
            n_generations: Number of evolutionary generations.
            verbose: Print progress every 10 generations.

        Returns:
            List of Pareto-optimal ``TreeGenotype`` instances.
        """
        # --- LDD-9: reproducibility ---
        if self.random_state is not None:
            random.seed(self.random_state)
            np.random.seed(self.random_state)

        # Initialize population
        population = self._create_population(X, y, population_size)

        # Evaluate initial population
        self._evaluate(population, X, y)

        # --- LDD-16: assign crowding distance before first DCD selection ---
        fronts = tools.sortNondominated(population, len(population))
        for front in fronts:
            tools.assignCrowdingDist(front)

        # Evolution loop
        for gen in range(n_generations):
            # Select offspring via tournament with crowding-distance comparison
            offspring = tools.selTournamentDCD(population, len(population))
            offspring = [self.toolbox.clone(ind) for ind in offspring]

            # --- LDD-2: real crossover ---
            for i in range(0, len(offspring) - 1, 2):
                if random.random() < self.crossover_prob:
                    tree1 = offspring[i][0]  # Individual wraps a TreeGenotype
                    tree2 = offspring[i + 1][0]
                    child1, child2 = safe_subtree_crossover(tree1, tree2)
                    offspring[i] = self._wrap_individual(child1)
                    offspring[i + 1] = self._wrap_individual(child2)
                    # Invalidate fitness
                    del offspring[i].fitness.values
                    del offspring[i + 1].fitness.values

            # --- LDD-2: real mutation ---
            for ind in offspring:
                if random.random() < self.mutation_prob:
                    mutated = self.mutation_fn(ind[0])
                    ind[0] = mutated
                    del ind.fitness.values

            # Evaluate offspring that need it
            invalid = [ind for ind in offspring if not ind.fitness.valid]
            self._evaluate(invalid, X, y)

            # NSGA-II environmental selection (LDD-16: assigns crowding dist)
            population = tools.selNSGA2(population + offspring, population_size)

            if verbose and gen % 10 == 0:
                front0 = tools.sortNondominated(population, len(population), first_front_only=True)[
                    0
                ]
                print(f"Gen {gen}: front size={len(front0)}, pop={len(population)}")

        # Extract Pareto front
        pareto_front = tools.sortNondominated(population, len(population), first_front_only=True)[0]

        # Unwrap TreeGenotype objects
        return [ind[0] for ind in pareto_front]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _create_population(self, X: np.ndarray, y: np.ndarray, size: int) -> list:
        """Create initial DEAP-wrapped population."""
        population = []
        for _ in range(size):
            tree = self.initializer.create_random_tree(X, y)
            ind = self._wrap_individual(tree)
            population.append(ind)
        return population

    @staticmethod
    def _wrap_individual(tree: TreeGenotype):
        """Wrap a TreeGenotype in a DEAP Individual."""
        ind = creator.Individual([tree])
        return ind

    def _evaluate(self, population: list, X: np.ndarray, y: np.ndarray) -> None:
        """Evaluate fitness for individuals that need it."""
        for ind in population:
            if not ind.fitness.valid:
                tree = ind[0]
                accuracy, interpretability = self.fitness_fn(tree, X, y)
                ind.fitness.values = (accuracy, interpretability)

    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------

    @staticmethod
    def plot_pareto_front(
        pareto_front: List[TreeGenotype], save_path: Optional[str] = None
    ) -> None:
        """Visualize the Pareto front.

        Args:
            pareto_front: List of Pareto-optimal trees (must have
                ``accuracy_`` and ``interpretability_`` attributes set).
            save_path: Optional file path to save the figure.
        """
        import matplotlib.pyplot as plt

        accuracies = [t.accuracy_ for t in pareto_front]
        interpretabilities = [t.interpretability_ for t in pareto_front]

        plt.figure(figsize=(10, 8))
        plt.scatter(
            interpretabilities,
            accuracies,
            s=100,
            alpha=0.6,
            c=range(len(pareto_front)),
            cmap="viridis",
        )
        plt.xlabel("Interpretability Score", fontsize=12)
        plt.ylabel("Accuracy", fontsize=12)
        plt.title("Pareto Front: Accuracy vs Interpretability", fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.colorbar(label="Solution Index")

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.show()
