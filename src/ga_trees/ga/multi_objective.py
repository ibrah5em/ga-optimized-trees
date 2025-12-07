"""
Multi-objective optimization with NSGA-II.

This implements Pareto-based optimization for accuracy vs interpretability.
"""

import random
from typing import List, Tuple

import numpy as np
from deap import algorithms, base, creator, tools

from ga_trees.genotype.tree_genotype import TreeGenotype


class ParetoOptimizer:
    """NSGA-II optimizer for multi-objective tree evolution."""

    def __init__(self, initializer, fitness_calculators: dict):
        """
        Initialize Pareto optimizer.

        Args:
            initializer: TreeInitializer instance
            fitness_calculators: Dict with 'accuracy' and 'interpretability' calculators
        """
        self.initializer = initializer
        self.fitness_calculators = fitness_calculators

        # Setup DEAP
        if not hasattr(creator, "FitnessMulti"):
            creator.create("FitnessMulti", base.Fitness, weights=(1.0, 1.0))  # Maximize both
        if not hasattr(creator, "Individual"):
            creator.create("Individual", TreeGenotype, fitness=creator.FitnessMulti)

        self.toolbox = base.Toolbox()

    def evaluate_multi_objective(self, tree: TreeGenotype, X, y) -> Tuple[float, float]:
        """Calculate both objectives."""
        accuracy = self.fitness_calculators["accuracy"](tree, X, y)
        interpretability = self.fitness_calculators["interpretability"](tree)
        return (accuracy, interpretability)

    def evolve_pareto_front(
        self, X, y, population_size=100, n_generations=50
    ) -> List[TreeGenotype]:
        """
        Evolve Pareto-optimal solutions.

        Returns:
            List of Pareto-optimal trees
        """
        # Initialize population
        population = [self.initializer.create_random_tree(X, y) for _ in range(population_size)]

        # Evaluate initial population
        fitnesses = [self.evaluate_multi_objective(ind, X, y) for ind in population]
        for ind, fit in zip(population, fitnesses):
            ind.fitness.values = fit

        # Evolution loop with NSGA-II
        for gen in range(n_generations):
            # Select offspring
            offspring = tools.selTournamentDCD(population, len(population))
            offspring = [self.toolbox.clone(ind) for ind in offspring]

            # Apply crossover and mutation
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < 0.7:
                    # Crossover logic here
                    pass

            for mutant in offspring:
                if random.random() < 0.2:
                    # Mutation logic here
                    pass

            # Evaluate offspring
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = [self.evaluate_multi_objective(ind, X, y) for ind in invalid_ind]
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            # Select next generation using NSGA-II
            population = tools.selNSGA2(population + offspring, population_size)

        # Extract Pareto front
        pareto_front = tools.sortNondominated(population, len(population), first_front_only=True)[0]

        return list(pareto_front)

    def plot_pareto_front(self, pareto_front: List[TreeGenotype], save_path: str = None):
        """Visualize Pareto front."""
        import matplotlib.pyplot as plt

        accuracies = [tree.accuracy_ for tree in pareto_front]
        interpretabilities = [tree.interpretability_ for tree in pareto_front]

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
