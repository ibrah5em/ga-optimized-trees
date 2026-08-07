"""Nested cross-validation benchmark harness.

Separate from ``scripts/experiment.py``, which runs a flat single-level CV and
is kept for quick screening. Anything reported in the paper comes from here.

Two harnesses live here and they answer different questions:

* :mod:`~ga_trees.benchmark.nested_cv` reports one tuned operating point per
  method per fold — accuracy, leaf count, path length. This is what H2/K3 (TOST
  equivalence against inner-CV-tuned CART) is written against.
* :mod:`~ga_trees.benchmark.frontiers` reports a *frontier* per method per fold
  and scores it by hypervolume. **K1 and H1/K2 are stated on hypervolume**, so
  the point-estimate harness alone cannot decide them.
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
