"""GA package exports.

Expose genetic algorithm engine and operators for easy import.
"""

from .engine import Crossover, GAConfig, GAEngine, Mutation, Selection, TreeInitializer
from .improved_crossover import safe_subtree_crossover

__all__ = [
    "GAEngine",
    "GAConfig",
    "TreeInitializer",
    "Selection",
    "Crossover",
    "Mutation",
    "safe_subtree_crossover",
]

# ParetoOptimizer requires DEAP (optional dependency)
try:
    from .multi_objective import ParetoOptimizer  # noqa: F401

    __all__.append("ParetoOptimizer")
except ImportError:
    pass
