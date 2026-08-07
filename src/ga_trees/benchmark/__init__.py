"""Nested cross-validation benchmark harness.

Separate from ``scripts/experiment.py``, which runs a flat single-level CV and
is kept for quick screening. Anything reported in the paper comes from here.
"""

from .methods import (
    GATreeMethod,
    PrunedCARTMethod,
    RandomForestMethod,
    RandomTreeSearch,
    UnconstrainedCARTMethod,
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
    "results_to_nested_dict",
    "run_nested_cv",
    "select_hyperparameters",
    "verify_budget_match",
]
