"""Nested cross-validation benchmark harness.

Separate from ``scripts/experiment.py``, which runs a flat single-level CV and
is kept for quick screening. Reported benchmark results come from here.

Two harnesses live here and they answer different questions:

* :mod:`~ga_trees.benchmark.nested_cv` reports one tuned operating point per
  method per fold — accuracy, leaf count, path length. This is what the TOST
  equivalence check against inner-CV-tuned CART is written against.
* :mod:`~ga_trees.benchmark.frontiers` reports a *frontier* per method per fold
  and scores it by hypervolume. **The comparisons against random search and
  CART's pruning path are stated on hypervolume**, so the point-estimate harness
  alone cannot decide them.
"""

from .frontiers import (
    CARTPathFrontier,
    FrontierFoldResult,
    FrontierMethod,
    ParetoGAFrontier,
    RandomSearchFrontier,
    dominance_rate,
    hypervolume_by_dataset,
    run_frontier_cv,
)
from .methods import (
    GATreeMethod,
    PrunedCARTMethod,
    RandomForestMethod,
    RandomTreeSearch,
    UnconstrainedCARTMethod,
    holdout_split,
)
from .nested_cv import (
    FoldResult,
    results_to_nested_dict,
    run_nested_cv,
    select_hyperparameters,
    verify_budget_match,
)
from .protocol import BenchmarkMethod, FittedModel

__all__ = [
    "BenchmarkMethod",
    "FittedModel",
    "FoldResult",
    "GATreeMethod",
    "PrunedCARTMethod",
    "RandomForestMethod",
    "RandomTreeSearch",
    "UnconstrainedCARTMethod",
    "holdout_split",
    "results_to_nested_dict",
    "run_nested_cv",
    "select_hyperparameters",
    "verify_budget_match",
    "CARTPathFrontier",
    "FrontierFoldResult",
    "FrontierMethod",
    "ParetoGAFrontier",
    "RandomSearchFrontier",
    "dominance_rate",
    "hypervolume_by_dataset",
    "run_frontier_cv",
]
