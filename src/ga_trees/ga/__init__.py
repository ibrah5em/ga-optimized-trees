"""GA package exports.

Expose genetic algorithm engine and operators for easy import.
"""

from .engine import Crossover, GAConfig, GAEngine, Mutation, Selection, TreeInitializer
from .improved_crossover import safe_subtree_crossover
from .split_points import (
    MIDPOINT_STRATEGY,
    UNIFORM_STRATEGY,
    candidate_thresholds,
    sample_threshold,
)

__all__ = [
    "GAEngine",
    "GAConfig",
    "TreeInitializer",
    "Selection",
    "Crossover",
    "Mutation",
    "safe_subtree_crossover",
    "candidate_thresholds",
    "sample_threshold",
    "MIDPOINT_STRATEGY",
    "UNIFORM_STRATEGY",
]

# ParetoOptimizer requires DEAP (optional dependency)
try:
    from .multi_objective import ParetoOptimizer  # noqa: F401

    __all__.append("ParetoOptimizer")
except ImportError:
    pass
