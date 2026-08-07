"""Frontier-level benchmark: the harness K1 and H1 are actually written against.

``benchmark/nested_cv.py`` reports one operating point per method per fold, which
answers "is the GA's chosen tree as accurate as CART's" but not the
pre-registered questions. Both K1 and H1 are stated on **hypervolume**:

    K1 — if budget-matched random search matches the GA on hypervolume, the
         evolutionary machinery contributes nothing.
    H1 — the evolved frontier dominates the one from CART's ccp_alpha path.

A hypervolume needs a *set* of models per fold, so this module runs each method
in a mode that produces one:

* :class:`ParetoGAFrontier` — NSGA-II over **(accuracy, −node count)**, which is
  Phase 2 item 6. The shipped objective pair was (accuracy, composite
  interpretability) on resubstitution data; the composite score may not be a
  reported outcome under K4, and hypervolume against it would not be the
  pre-registered measurement.
* :class:`RandomSearchFrontier` — the same tree space and the same number of
  evaluations, keeping its whole non-dominated set rather than a single best.
  Giving random search only its best point would compare a frontier against a
  point and guarantee the GA wins, which is not a test.
* :class:`CARTPathFrontier` — the cost-complexity pruning path, H1's comparator.

Budget matching is done by *measurement, not prediction*: the GA runs first, its
realised evaluation count is read off a counter, and random search is then given
exactly that many draws. NSGA-II's per-generation cost depends on how many
offspring crossover and mutation actually invalidated, so any closed-form budget
would be wrong by a variable margin.

**Every method delivers a train-selected set, scored on test afterwards.** This
is not cosmetic. An earlier version of this module returned all ~2,500 of random
search's sampled trees and let the dominance filter run on their *test* scores,
which is the maximum over thousands of test evaluations — a quantity no method
can actually deliver, because choosing among those candidates needs the test
labels. It inflated random search's hypervolume and produced a K1 result that
was an artefact of the harness. Both searchers now maintain a non-dominated
archive on training/validation objectives and hand over only that.

**One asymmetry remains, and it is reported rather than resolved.** NSGA-II's
natural answer is the front of its *final population*; random search has no
population, so its answer is necessarily an archive. A point the GA found in
generation 3 and lost by generation 20 counts for random search's analogue but
not for the GA. :class:`ParetoGAFrontier` therefore takes an ``archive`` flag:

* ``archive=False`` — the final-population front. The pre-registered reading.
* ``archive=True`` — the same run's non-dominated archive, exactly the
  bookkeeping random search gets.

Run both. The difference between them is how much of any gap is the search and
how much is the bookkeeping, and reporting only whichever is more flattering
would be the failure this whole plan exists to correct.
"""

import abc
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold

from ga_trees.benchmark.methods import DEFAULT_VALIDATION_FRACTION, holdout_split
from ga_trees.evaluation.hypervolume import frontier, hypervolume, reference_nodes_for
from ga_trees.fitness.calculator import FitnessCalculator, TreePredictor
from ga_trees.ga.engine import Mutation, TreeInitializer
from ga_trees.ga.split_points import MIDPOINT_STRATEGY
from ga_trees.reproducibility import derive_fold_seed

logger = logging.getLogger(__name__)

#: Added to the largest node count seen on a dataset to form the shared reference
#: point, so the largest model still contributes non-zero area.
#:
#: ``paper/PREREGISTRATION.md`` fixes the reference at "nodes = max over all
#: methods on that **dataset**", so it is computed once per dataset over every
#: fold, not per fold. The distinction is not cosmetic: a per-fold reference is a
#: smaller box, and since raising the reference adds ``max_accuracy x delta`` to a
#: method's area, a smaller box favours whichever method has the lower peak
#: accuracy. Measured on the real run it moved the K2 dominance rate from 45% to
#: 70% — across the 60% threshold, in the GA's favour. The pre-registered
#: definition is the one used.
REFERENCE_MARGIN = 1.0


@dataclass
class FrontierFoldResult:
    """One (dataset, method, outer fold) frontier."""

    dataset: str
    method: str
    fold: int
    seed: int
    hypervolume: float
    n_points: int
    n_distinct: int
    n_candidates: int
    n_evaluations: int
    reference_nodes: float
    fit_seconds: float
    best_accuracy: float
    smallest_nodes: float
    points: List[Tuple[float, float]] = field(default_factory=list)

    def as_row(self) -> Dict[str, Any]:
        """Flatten for CSV. The point list is kept out; it goes to its own file."""
        return {
            "dataset": self.dataset,
            "method": self.method,
            "fold": self.fold,
            "seed": self.seed,
            "hypervolume": self.hypervolume,
            "n_points": self.n_points,
            "n_distinct": self.n_distinct,
            "n_candidates": self.n_candidates,
            "n_evaluations": self.n_evaluations,
            "reference_nodes": self.reference_nodes,
            "fit_seconds": self.fit_seconds,
            "best_accuracy": self.best_accuracy,
            "smallest_nodes": self.smallest_nodes,
        }


class _CountingObjective:
    """Two-objective fitness wrapper that counts evaluations.

    With ``archive=True`` it also keeps the non-dominated set over every
    candidate it ever scored. Archiving is done on the *training/validation*
    objectives, never on test data, so it is a property of the search and not a
    selection step.
    """

    def __init__(self, calculator: FitnessCalculator, X_val, y_val, archive: bool = False):
        self.calculator = calculator
        self.X_val = X_val
        self.y_val = y_val
        self.count = 0
        self.archiving = archive
        self.archive: List[Tuple[float, float, Any]] = []

    def __call__(self, tree, X, y) -> Tuple[float, float]:
        self.count += 1
        if self.X_val is not None:
            self.calculator.calculate_fitness(tree, X, y, self.X_val, self.y_val)
        else:
            self.calculator.calculate_fitness(tree, X, y)
        # Maximise accuracy, minimise size. ParetoOptimizer maximises both
        # objectives, so size enters negated rather than as the composite
        # interpretability score (Phase 2 item 6, and K4).
        objectives = (float(tree.accuracy_), -float(tree.get_num_nodes()))
        if self.archiving:
            self._offer(objectives, tree)
        return objectives

    def _offer(self, objectives: Tuple[float, float], tree) -> None:
        """Add *tree* to the archive if nothing in it already dominates *tree*."""
        accuracy, negated_nodes = objectives
        for archived_accuracy, archived_nodes, _ in self.archive:
            dominates = archived_accuracy >= accuracy and archived_nodes >= negated_nodes
            strict = archived_accuracy > accuracy or archived_nodes > negated_nodes
            if dominates and strict:
                return
            if archived_accuracy == accuracy and archived_nodes == negated_nodes:
                return  # same objective vector; one representative is enough

        self.archive = [
            entry
            for entry in self.archive
            if not (
                accuracy >= entry[0]
                and negated_nodes >= entry[1]
                and (accuracy > entry[0] or negated_nodes > entry[1])
            )
        ]
        # Copied because NSGA-II reuses and rewrites individuals across
        # generations; storing the live object would archive a moving target.
        self.archive.append((accuracy, negated_nodes, tree.copy()))

    def archived_trees(self) -> List[Any]:
        return [tree for _, _, tree in self.archive]


class FrontierMethod(abc.ABC):
    """A method that produces a set of models, not one model."""

    name = "unnamed"

    @abc.abstractmethod
    def build(self, X: np.ndarray, y: np.ndarray, seed: int) -> Tuple[List, int]:
        """Return ``(candidate trees, evaluations spent)`` fitted on ``(X, y)``."""

    def evaluation_budget(self) -> Optional[int]:
        """Evaluations this method spent on its last :meth:`build`."""
        return None


def _feature_ranges(X: np.ndarray) -> Dict[int, tuple]:
    return {j: (float(X[:, j].min()), float(X[:, j].max())) for j in range(X.shape[1])}


def _make_search_pieces(tree_config, fitness_config, X, y, seed, archive: bool = False):
    """Initializer, mutation, calculator and data split, shared by both searchers."""
    X_train, y_train, X_val, y_val = holdout_split(
        X,
        y,
        fraction=float(fitness_config.get("validation_fraction", DEFAULT_VALIDATION_FRACTION)),
        seed=seed,
    )
    initializer = TreeInitializer(
        n_features=X.shape[1],
        n_classes=len(np.unique(y)),
        max_depth=tree_config["max_depth"],
        min_samples_split=tree_config["min_samples_split"],
        min_samples_leaf=tree_config["min_samples_leaf"],
        growth_stop_prob=tree_config.get("growth_stop_prob", 0.3),
        split_strategy=tree_config.get("split_strategy", MIDPOINT_STRATEGY),
    )
    mutation = Mutation(
        n_features=X.shape[1],
        feature_ranges=_feature_ranges(X_train),
        X=X_train,
        min_samples_leaf=tree_config["min_samples_leaf"],
        split_strategy=tree_config.get("split_strategy", MIDPOINT_STRATEGY),
    )
    calculator = FitnessCalculator(
        mode="weighted_sum",  # scalar mode; the objective pair is built below
        classification_metric=fitness_config.get("classification_metric", "accuracy"),
    )
    objective = _CountingObjective(calculator, X_val, y_val, archive=archive)
    return initializer, mutation, objective, X_train, y_train


class ParetoGAFrontier(FrontierMethod):
    """NSGA-II over (accuracy, −node count).

    Parameters
    ----------
    archive : bool
        When False (the pre-registered measurement) the frontier is the front of
        the *final population*. When True it is the non-dominated set over every
        candidate the run ever scored, which is what random search implicitly
        gets. Run both: the difference between them is how much of any gap is
        the algorithm and how much is the bookkeeping.
    """

    def __init__(self, ga_config, tree_config, fitness_config, archive: bool = False):
        self.ga_config = dict(ga_config)
        self.tree_config = dict(tree_config)
        self.fitness_config = dict(fitness_config)
        self.archive = archive
        self.name = "GA (NSGA-II, archived)" if archive else "GA (NSGA-II)"

    def build(self, X, y, seed):
        from ga_trees.ga.multi_objective import ParetoOptimizer

        initializer, mutation, objective, X_train, y_train = _make_search_pieces(
            self.tree_config, self.fitness_config, X, y, seed, archive=self.archive
        )
        optimizer = ParetoOptimizer(
            initializer=initializer,
            fitness_fn=objective,
            mutation_fn=lambda tree: mutation.mutate(tree, self.ga_config.get("mutation_types")),
            crossover_prob=self.ga_config["crossover_prob"],
            mutation_prob=self.ga_config["mutation_prob"],
            random_state=seed,
        )
        front = optimizer.evolve_pareto_front(
            X_train,
            y_train,
            population_size=self.ga_config["population_size"],
            n_generations=self.ga_config["n_generations"],
            verbose=False,
        )
        if self.archive:
            return objective.archived_trees(), objective.count
        return front, objective.count


class RandomSearchFrontier(FrontierMethod):
    """Budget-matched random sampling, delivering its non-dominated archive.

    The archive is maintained on the **training/validation** objectives, and
    only the archive is later scored on the test fold. Returning every sampled
    candidate instead would hand random search the maximum over thousands of
    test evaluations — an optimistically biased quantity that no method could
    actually deliver, since choosing among those candidates requires the test
    labels. That is a selection-on-test effect, not a frontier.
    """

    name = "Random Search"

    def __init__(self, ga_config, tree_config, fitness_config):
        self.ga_config = dict(ga_config)
        self.tree_config = dict(tree_config)
        self.fitness_config = dict(fitness_config)
        self.budget: Optional[int] = None

    def build(self, X, y, seed):
        import random as _random

        if self.budget is None:
            raise ValueError(
                "RandomSearchFrontier.budget must be set from the GA's realised "
                "evaluation count before build(); K1 is meaningless otherwise."
            )

        initializer, _, objective, X_train, y_train = _make_search_pieces(
            self.tree_config, self.fitness_config, X, y, seed, archive=True
        )
        _random.seed(seed)
        np.random.seed(seed)

        for _ in range(self.budget):
            tree = initializer.create_random_tree(X_train, y_train)
            objective(tree, X_train, y_train)
        return objective.archived_trees(), objective.count


class CARTPathFrontier(FrontierMethod):
    """Cost-complexity pruning path — H1's comparator."""

    name = "CART (ccp path)"

    def __init__(self, tree_config, max_alphas: int = 12):
        self.tree_config = dict(tree_config)
        self.max_alphas = max_alphas

    def build(self, X, y, seed):
        from sklearn.tree import DecisionTreeClassifier

        kwargs = dict(
            min_samples_split=self.tree_config["min_samples_split"],
            min_samples_leaf=self.tree_config["min_samples_leaf"],
        )
        probe = DecisionTreeClassifier(random_state=seed, **kwargs)
        try:
            alphas = np.unique(probe.cost_complexity_pruning_path(X, y).ccp_alphas)
            alphas = alphas[alphas >= 0]
        except (ValueError, AttributeError):
            alphas = np.array([0.0])
        if len(alphas) > self.max_alphas:
            alphas = alphas[np.linspace(0, len(alphas) - 1, self.max_alphas).astype(int)]

        estimators = []
        for alpha in alphas:
            estimator = DecisionTreeClassifier(ccp_alpha=float(alpha), random_state=seed, **kwargs)
            estimator.fit(X, y)
            estimators.append(estimator)
        # Pruning is a deterministic sweep, not a search: no candidates are
        # evaluated and discarded, so there is no budget to match.
        return estimators, 0


def _score_candidates(
    candidates: Sequence, X_train, y_train, X_test, y_test
) -> List[Tuple[float, float]]:
    """Test-set (accuracy, node count) for each candidate model.

    Candidates are chosen on training data; this only measures them. Filtering
    for dominance afterwards is a property of the point set, not a selection
    step, so it does not leak the test fold.
    """
    predictor = TreePredictor()
    points = []
    for candidate in candidates:
        if hasattr(candidate, "tree_"):  # fitted sklearn estimator
            accuracy = float(np.mean(candidate.predict(X_test) == y_test))
            nodes = float(candidate.tree_.node_count)
        else:  # TreeGenotype
            predictor.fit_leaf_predictions(candidate, X_train, y_train)
            accuracy = float(np.mean(predictor.predict(candidate, X_test) == y_test))
            nodes = float(candidate.get_num_nodes())
        points.append((accuracy, nodes))
    return points


def run_frontier_cv(
    X: np.ndarray,
    y: np.ndarray,
    ga_method: ParetoGAFrontier,
    random_method: RandomSearchFrontier,
    other_methods: Sequence[FrontierMethod],
    dataset_name: str,
    base_seed: int = 42,
    outer_splits: int = 10,
    outer_repeats: int = 3,
    progress: Optional[Callable[[str], None]] = None,
) -> List[FrontierFoldResult]:
    """Outer CV producing one frontier per method per fold.

    The GA runs first on each fold so that its realised evaluation count can be
    handed to random search, which is the only way to budget-match a variable
    cost exactly.

    Returns
    -------
    list of FrontierFoldResult
        Hypervolumes are comparable within a (dataset, fold) — every method on a
        fold is scored against one shared reference point.
    """
    _, counts = np.unique(y, return_counts=True)
    if counts.min() < outer_splits:
        logger.warning(
            "%s: smallest class has %d members but outer_splits=%d; reducing.",
            dataset_name,
            counts.min(),
            outer_splits,
        )
        outer_splits = max(2, int(counts.min()))

    outer = RepeatedStratifiedKFold(
        n_splits=outer_splits, n_repeats=outer_repeats, random_state=base_seed
    )

    # Pass 1 — build every fold's frontier. Hypervolume is deferred because the
    # reference point is defined over the whole dataset, so it is not known until
    # every fold has been seen.
    pending: List[dict] = []
    ordered: List[FrontierMethod] = [ga_method, random_method, *other_methods]

    for fold, (train_idx, test_idx) in enumerate(outer.split(X, y), 1):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # GA first — random search's budget is read from what it actually spent.
        for method in ordered:
            seed = derive_fold_seed(base_seed, dataset_name, fold, method.name)
            started = time.time()
            candidates, evaluations = method.build(X_train, y_train, seed)
            elapsed = time.time() - started

            if method is ga_method:
                random_method.budget = evaluations

            points = _score_candidates(candidates, X_train, y_train, X_test, y_test)
            pending.append(
                {
                    "fold": fold,
                    "method": method.name,
                    "seed": seed,
                    "front": frontier(points),
                    "candidates": len(candidates),
                    "evaluations": evaluations,
                    "elapsed": elapsed,
                }
            )

    # Pass 2 — one reference point for the dataset, per the pre-registration.
    reference = reference_nodes_for(
        {f"{e['fold']}:{e['method']}": e["front"] for e in pending}, margin=REFERENCE_MARGIN
    )

    results: List[FrontierFoldResult] = []
    for entry in pending:
        front = entry["front"]
        volume = hypervolume(front, reference_nodes=reference)
        results.append(
            FrontierFoldResult(
                dataset=dataset_name,
                method=entry["method"],
                fold=entry["fold"],
                seed=entry["seed"],
                hypervolume=volume,
                n_points=len(front),
                n_distinct=front.n_distinct,
                n_candidates=entry["candidates"],
                n_evaluations=entry["evaluations"],
                reference_nodes=reference,
                fit_seconds=entry["elapsed"],
                best_accuracy=float(front.accuracies.max()) if len(front) else 0.0,
                smallest_nodes=float(front.node_counts.min()) if len(front) else 0.0,
                points=[(float(a), float(n)) for a, n in front.points],
            )
        )
        if progress is not None:
            progress(
                f"  {dataset_name} fold {entry['fold']:>3} {entry['method']:16s} "
                f"hv={volume:>9.2f} pts={len(front):>3} "
                f"distinct={front.n_distinct:>4} ({entry['elapsed']:.1f}s)"
            )

    return results


def hypervolume_by_dataset(
    results: Sequence[FrontierFoldResult],
) -> Dict[str, Dict[str, List[float]]]:
    """Reshape to ``{dataset: {method: [per-fold hypervolume]}}``."""
    nested: Dict[str, Dict[str, List[float]]] = {}
    for result in results:
        nested.setdefault(result.dataset, {}).setdefault(result.method, []).append(
            result.hypervolume
        )
    return nested


def dominance_rate(results: Sequence[FrontierFoldResult], method: str, baseline: str) -> float:
    """Fraction of datasets where *method* has the larger mean hypervolume.

    K2 rejects H1 below 60%.
    """
    nested = hypervolume_by_dataset(results)
    wins, total = 0, 0
    for methods in nested.values():
        if method not in methods or baseline not in methods:
            continue
        total += 1
        if float(np.mean(methods[method])) > float(np.mean(methods[baseline])):
            wins += 1
    return wins / total if total else 0.0
