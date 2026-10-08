"""Nested cross-validation harness.

The protocol: an outer 10-fold × 3-repeat
stratified CV for reporting, and an inner 5-fold CV for *all* hyperparameter
selection, applied identically to every method.

Why nested. Selecting hyperparameters on the same folds used for reporting
leaks the test set into model selection and inflates the reported score — the
optimistic bias is well documented (Varma & Simon 2006; Cawley & Talbot 2010).
The previous protocol here selected nothing but also tuned nothing, so the GA
ran at hand-picked settings while CART ran at a fixed ``max_depth=6``. Nested CV
removes that asymmetry: every method gets the same tuning opportunity on data
the outer fold never sees.

Cost. Outer folds × repeats × inner folds × grid size fits per method. With the
default 10 × 3 × 5 that is 150 fits per grid point, so grids are kept small and
the GA's population and generation counts are held fixed rather than tuned.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold

from ga_trees.benchmark.protocol import BenchmarkMethod
from ga_trees.reproducibility import derive_fold_seed

logger = logging.getLogger(__name__)

#: Outer protocol, fixed before the benchmark was run.
DEFAULT_OUTER_SPLITS = 10
DEFAULT_OUTER_REPEATS = 3
DEFAULT_INNER_SPLITS = 5


@dataclass
class FoldResult:
    """One (dataset, method, outer fold) cell."""

    dataset: str
    method: str
    fold: int
    seed: int
    test_accuracy: float
    test_f1: float
    fit_seconds: float
    selected_params: Dict[str, Any] = field(default_factory=dict)
    complexity: Dict[str, float] = field(default_factory=dict)

    def as_row(self) -> Dict[str, Any]:
        """Flatten to a CSV-friendly row."""
        row = {
            "dataset": self.dataset,
            "method": self.method,
            "fold": self.fold,
            "seed": self.seed,
            "test_accuracy": self.test_accuracy,
            "test_f1": self.test_f1,
            "fit_seconds": self.fit_seconds,
            "selected_params": ";".join(
                f"{k}={v}" for k, v in sorted(self.selected_params.items())
            ),
        }
        row.update(self.complexity)
        return row


def _score(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float]:
    return (
        float(accuracy_score(y_true, y_pred)),
        float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    )


def select_hyperparameters(
    method: BenchmarkMethod,
    X: np.ndarray,
    y: np.ndarray,
    seed: int,
    inner_splits: int = DEFAULT_INNER_SPLITS,
    scorer: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
) -> Dict[str, Any]:
    """Choose hyperparameters by inner CV on the outer-training data alone.

    Parameters
    ----------
    method : BenchmarkMethod
        Method to tune.
    X, y : ndarray
        Outer *training* split. The outer test split must never reach here.
    seed : int
        Seed for the inner splitter and for each candidate fit.
    inner_splits : int
        Inner fold count.
    scorer : callable, optional
        ``scorer(y_true, y_pred) -> float``, higher is better. Defaults to
        accuracy.

    Returns
    -------
    dict
        The winning parameter dict. Ties resolve to the first candidate, so
        grids should be ordered simplest-first.
    """
    scorer = scorer or (lambda a, b: float(accuracy_score(a, b)))
    grid = method.param_grid(X, y)
    if len(grid) == 1:
        return dict(grid[0])

    # Stratified inner CV needs at least as many members per class as folds.
    _, counts = np.unique(y, return_counts=True)
    usable_splits = int(min(inner_splits, counts.min()))
    if usable_splits < 2:
        logger.warning(
            "Smallest class has %d member(s); inner CV not possible, using first grid point.",
            counts.min(),
        )
        return dict(grid[0])

    inner = StratifiedKFold(n_splits=usable_splits, shuffle=True, random_state=seed)
    splits = list(inner.split(X, y))

    best_params, best_score = dict(grid[0]), -np.inf
    for candidate in grid:
        scores = []
        for inner_fold, (train_idx, val_idx) in enumerate(splits):
            model = method.fit(X[train_idx], y[train_idx], candidate, seed + inner_fold + 1)
            scores.append(scorer(y[val_idx], model.predict(X[val_idx])))
        mean_score = float(np.mean(scores))
        if mean_score > best_score:
            best_score = mean_score
            best_params = dict(candidate)

    return best_params


def run_nested_cv(
    X: np.ndarray,
    y: np.ndarray,
    methods: Sequence[BenchmarkMethod],
    dataset_name: str,
    base_seed: int = 42,
    outer_splits: int = DEFAULT_OUTER_SPLITS,
    outer_repeats: int = DEFAULT_OUTER_REPEATS,
    inner_splits: int = DEFAULT_INNER_SPLITS,
    progress: Optional[Callable[[str], None]] = None,
) -> List[FoldResult]:
    """Run the full nested protocol for one dataset.

    Every method sees identical outer splits, so comparisons are paired. Seeds
    come from :func:`derive_fold_seed`, which mixes in the method name — two
    methods on the same fold must not share a random stream, or their results
    become correlated through the RNG.

    Parameters
    ----------
    X, y : ndarray
        Full dataset. Splitting happens here, not before.
    methods : sequence of BenchmarkMethod
        Methods to compare.
    dataset_name : str
        Used in result rows and in seed derivation.
    base_seed : int
        Experiment-wide seed.
    outer_splits, outer_repeats, inner_splits : int
        Protocol sizes. Defaults are the benchmark's values.
    progress : callable, optional
        Called with a status line after each (fold, method) cell.

    Returns
    -------
    list of FoldResult
        One entry per (method, outer fold).
    """
    if len(methods) == 0:
        raise ValueError("run_nested_cv() requires at least one method.")

    _, counts = np.unique(y, return_counts=True)
    if counts.min() < outer_splits:
        logger.warning(
            "%s: smallest class has %d members but outer_splits=%d; reducing to %d.",
            dataset_name,
            counts.min(),
            outer_splits,
            counts.min(),
        )
        outer_splits = max(2, int(counts.min()))

    outer = RepeatedStratifiedKFold(
        n_splits=outer_splits, n_repeats=outer_repeats, random_state=base_seed
    )

    results: List[FoldResult] = []
    # Folds are 1-indexed to match build_seed_manifest, which enumerates
    # 1..n_folds. A 0-indexed loop here would write a manifest whose seeds do
    # not correspond to the run it claims to document.
    for fold, (train_idx, test_idx) in enumerate(outer.split(X, y), 1):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        for method in methods:
            seed = derive_fold_seed(base_seed, dataset_name, fold, method.name)
            started = time.time()

            params = select_hyperparameters(
                method, X_train, y_train, seed=seed, inner_splits=inner_splits
            )
            model = method.fit(X_train, y_train, params, seed)
            accuracy, f1 = _score(y_test, model.predict(X_test))
            elapsed = time.time() - started

            results.append(
                FoldResult(
                    dataset=dataset_name,
                    method=method.name,
                    fold=fold,
                    seed=seed,
                    test_accuracy=accuracy,
                    test_f1=f1,
                    fit_seconds=elapsed,
                    selected_params=params,
                    complexity=model.complexity(),
                )
            )
            if progress is not None:
                progress(
                    f"  {dataset_name} fold {fold:>3} {method.name:24s} "
                    f"acc={accuracy:.4f} leaves={model.n_leaves:>4} ({elapsed:.1f}s)"
                )

    return results


def results_to_nested_dict(
    results: Sequence[FoldResult],
) -> Dict[str, Dict[str, Dict[str, List[float]]]]:
    """Reshape fold results into the structure the statistics layer expects.

    Returns
    -------
    dict
        ``{dataset: {method: {"test_acc": [...], "test_f1": [...], ...}}}``,
        consumable by :func:`ga_trees.evaluation.statistics.per_dataset_means`.
    """
    nested: Dict[str, Dict[str, Dict[str, List[float]]]] = {}
    for result in results:
        method_bucket = nested.setdefault(result.dataset, {}).setdefault(
            result.method,
            {"test_acc": [], "test_f1": [], "time": [], "nodes": [], "leaves": [], "depth": []},
        )
        method_bucket["test_acc"].append(result.test_accuracy)
        method_bucket["test_f1"].append(result.test_f1)
        method_bucket["time"].append(result.fit_seconds)
        method_bucket["nodes"].append(result.complexity.get("nodes", float("nan")))
        method_bucket["leaves"].append(result.complexity.get("leaves", float("nan")))
        method_bucket["depth"].append(result.complexity.get("depth", float("nan")))
    return nested


def verify_budget_match(
    results: Sequence[FoldResult], methods: Sequence[str], tolerance: float = 0.05
) -> Dict[str, Any]:
    """Check that the budget-matched methods really did equal work.

    Comparing the GA with random search only means something if they evaluated comparable
    numbers of candidates. This reports what they actually spent rather than
    trusting the configuration.

    Parameters
    ----------
    results : sequence of FoldResult
        Rows from :func:`run_nested_cv`.
    methods : sequence of str
        Method names that are supposed to be budget-matched.
    tolerance : float
        Allowed relative spread before the match is reported as failed.

    Returns
    -------
    dict
        ``mean_evaluations`` per method, plus ``matched`` and ``spread``.
    """
    means = {}
    for name in methods:
        evaluations = [r.complexity.get("evaluations", 0) for r in results if r.method == name]
        means[name] = float(np.mean(evaluations)) if evaluations else 0.0

    values = [v for v in means.values() if v > 0]
    if len(values) < 2:
        return {"mean_evaluations": means, "matched": False, "spread": float("nan")}

    spread = (max(values) - min(values)) / max(values)
    return {
        "mean_evaluations": means,
        "matched": bool(spread <= tolerance),
        "spread": float(spread),
    }
