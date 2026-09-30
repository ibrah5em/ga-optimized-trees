"""Benchmarked methods, all behind the same interface.

The budget-matched pair is :class:`GATreeMethod` and :class:`RandomTreeSearch`:
both draw from the same tree space, both are given the same number of candidate
evaluations, and both select by the same fitness. That comparison is kill
criterion K1 in the pre-registered protocol — if random search matches the GA,
the evolutionary machinery contributes nothing.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.tree import DecisionTreeClassifier

from ga_trees.benchmark.protocol import (
    BenchmarkMethod,
    FittedModel,
    ga_tree_complexity,
    sklearn_tree_complexity,
)
from ga_trees.fitness.calculator import FitnessCalculator, TreePredictor
from ga_trees.ga.engine import GAConfig, GAEngine, Mutation, TreeInitializer
from ga_trees.ga.repair import repair_from_config
from ga_trees.ga.split_points import MIDPOINT_STRATEGY

logger = logging.getLogger(__name__)

#: Depth grid shared by the constrained tree methods, so CART is not handicapped
#: relative to the GA by an arbitrary fixed depth.
DEFAULT_DEPTH_GRID = (3, 4, 5, 6, 8)

#: Cap on distinct ccp_alpha values taken from the pruning path. The full path
#: has one entry per merge and is far too long to tune over on large datasets.
MAX_CCP_ALPHAS = 12

#: Fraction of the fitting data held out to score fitness on. 0.0 reproduces the
#: pre-Phase-2 resubstitution fitness and is the code default so that library
#: users and the existing tests are not silently switched onto a different
#: objective; the shipped configs set it explicitly.
DEFAULT_VALIDATION_FRACTION = 0.0


class _CountingFitness:
    """Wraps a fitness function and counts calls.

    Budget matching is verified rather than assumed: both searching methods
    report how many candidates they actually evaluated, and the harness records
    it alongside the score.
    """

    def __init__(self, fitness_fn):
        self._fitness_fn = fitness_fn
        self.count = 0

    def __call__(self, tree, X, y, *validation):
        self.count += 1
        return self._fitness_fn(tree, X, y, *validation)


def _feature_ranges(X: np.ndarray) -> Dict[int, tuple]:
    return {j: (float(X[:, j].min()), float(X[:, j].max())) for j in range(X.shape[1])}


def holdout_split(
    X: np.ndarray, y: np.ndarray, fraction: float, seed: int
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """Carve a GA-validation split off the fitting data.

    Fitness was resubstitution: :meth:`FitnessCalculator.calculate_fitness` fits
    leaf predictions on the rows it then scores, so a tree that memorises the
    fitting set scores perfectly on it. The search therefore ranked individuals
    by how well they overfit, which is the opposite of what the outer fold
    measures.

    The split is stratified and drawn once per fit rather than per generation:
    a moving target would invalidate the cached fitness that elites carry across
    generations, and would add selection noise on top of the signal.

    Returns
    -------
    tuple
        ``(X_train, y_train, X_val, y_val)``. The validation pair is
        ``(None, None)`` when no split was requested or the data is too small to
        stratify one, in which case fitness falls back to resubstitution.
    """
    if fraction <= 0.0:
        return X, y, None, None
    if not 0.0 < fraction < 1.0:
        raise ValueError(f"validation_fraction must be in [0, 1), got {fraction}.")

    _, counts = np.unique(y, return_counts=True)
    n_classes = len(counts)
    n_val = int(round(len(y) * fraction))

    # StratifiedShuffleSplit needs at least one sample per class on both sides.
    if counts.min() < 2 or n_val < n_classes or (len(y) - n_val) < n_classes:
        logger.warning(
            "Cannot hold out %.0f%% of %d samples across %d classes; "
            "falling back to resubstitution fitness for this fit.",
            fraction * 100,
            len(y),
            n_classes,
        )
        return X, y, None, None

    splitter = StratifiedShuffleSplit(n_splits=1, test_size=fraction, random_state=seed)
    train_idx, val_idx = next(splitter.split(X, y))
    return X[train_idx], y[train_idx], X[val_idx], y[val_idx]


def ga_evaluation_budget(ga_config: Dict[str, Any]) -> int:
    """Fitness evaluations a GA run costs.

    Not ``population_size * n_generations``. The initial population is scored
    once, and thereafter elites carry their fitness across generations
    (``elitism_selection`` copies individuals, and ``evaluate_population`` skips
    anything already scored), so each generation only pays for its non-elite
    offspring.

    Getting this wrong breaks K1: budget-matching random search to the naive
    product handed it ~9% fewer evaluations than the GA in a smoke run, which
    would have quietly biased the comparison toward the GA.

    Early stopping can end a run below this figure, so treat it as the ceiling
    and read the realised counts from ``verify_budget_match``.
    """
    population_size = int(ga_config["population_size"])
    n_generations = int(ga_config["n_generations"])
    n_elite = int(float(ga_config.get("elitism_ratio", 0.0)) * population_size)
    return population_size + n_generations * (population_size - n_elite)


class _SearchContext:
    """The tree space, fitness and data split shared by the GA and random search.

    K1 asks whether the evolutionary machinery contributes anything over random
    sampling of the same space. Any asymmetry between the two — a different
    candidate distribution, a different fitness, a different validation split —
    would surface as an algorithmic effect. Building both from this one object
    makes that impossible by construction instead of by review.
    """

    def __init__(self, initializer, fitness, X_train, y_train, X_val, y_val):
        self.initializer = initializer
        self.fitness = fitness
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val

    @property
    def uses_validation(self) -> bool:
        return self.X_val is not None and self.y_val is not None

    def score(self, tree) -> float:
        """Fitness of *tree*: leaves fitted on the train split, scored on val."""
        if self.uses_validation:
            return self.fitness(tree, self.X_train, self.y_train, self.X_val, self.y_val)
        return self.fitness(tree, self.X_train, self.y_train)


def _build_search_context(
    tree_config: Dict[str, Any],
    fitness_config: Dict[str, Any],
    X: np.ndarray,
    y: np.ndarray,
    params: Dict[str, Any],
    seed: int,
) -> _SearchContext:
    """Assemble the initializer, fitness and data split for a searching method."""
    accuracy_weight = params.get(
        "accuracy_weight", fitness_config.get("weights", {}).get("accuracy", 0.7)
    )
    max_depth = params.get("max_depth", tree_config["max_depth"])

    X_train, y_train, X_val, y_val = holdout_split(
        X,
        y,
        fraction=float(fitness_config.get("validation_fraction", DEFAULT_VALIDATION_FRACTION)),
        seed=seed,
    )

    initializer = TreeInitializer(
        n_features=X.shape[1],
        n_classes=len(np.unique(y)),
        max_depth=max_depth,
        min_samples_split=tree_config["min_samples_split"],
        min_samples_leaf=tree_config["min_samples_leaf"],
        growth_stop_prob=tree_config.get("growth_stop_prob", 0.3),
        split_strategy=tree_config.get("split_strategy", MIDPOINT_STRATEGY),
    )
    calculator = FitnessCalculator(
        mode="weighted_sum",
        accuracy_weight=accuracy_weight,
        interpretability_weight=1.0 - accuracy_weight,
        interpretability_weights=fitness_config.get("interpretability_weights"),
        classification_metric=fitness_config.get("classification_metric", "accuracy"),
    )

    return _SearchContext(
        initializer=initializer,
        fitness=_CountingFitness(calculator.calculate_fitness),
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
    )


#: Accuracy weights the searching methods are tuned over.
ACCURACY_WEIGHT_GRID = (0.5, 0.7, 0.9)
#: Depths they are tuned over when depth tuning is on.
SEARCH_DEPTH_GRID = (4, 6, 8)


def search_grid(tune_depth: bool = True) -> List[Dict[str, Any]]:
    """The inner-CV grid shared by the GA and random search.

    One function, so the two budget-matched methods cannot drift onto different
    grids. With ``tune_depth=False`` only the accuracy weighting is tuned and
    depth stays at the configured ``tree.max_depth`` — the reduced K3 grid
    recorded as a deviation from the pre-registered protocol (2026-09-29).
    """
    depths = SEARCH_DEPTH_GRID if tune_depth else (None,)
    grid = []
    for accuracy_weight in ACCURACY_WEIGHT_GRID:
        for max_depth in depths:
            params = {"accuracy_weight": accuracy_weight}
            if max_depth is not None:
                params["max_depth"] = max_depth
            grid.append(params)
    return grid


class GATreeMethod(BenchmarkMethod):
    """The GA under test.

    Parameters
    ----------
    ga_config : dict
        The ``ga`` section of the experiment config.
    tree_config : dict
        The ``tree`` section.
    fitness_config : dict
        The ``fitness`` section.
    tune : bool
        When True, inner CV selects over accuracy/interpretability weightings.
        When False the configured weighting is used as-is, which is much
        cheaper and appropriate for screening runs.
    """

    name = "GA-Optimized"

    def __init__(
        self,
        ga_config: Dict[str, Any],
        tree_config: Dict[str, Any],
        fitness_config: Dict[str, Any],
        tune: bool = True,
        tune_depth: bool = True,
    ):
        self.ga_config = dict(ga_config)
        self.tree_config = dict(tree_config)
        self.fitness_config = dict(fitness_config)
        self.tune = tune
        self.tune_depth = tune_depth

    def param_grid(self, X: np.ndarray, y: np.ndarray) -> List[Dict[str, Any]]:
        if not self.tune:
            return [{}]
        # Only the accuracy/interpretability trade-off and depth are tuned.
        # Population and generation counts are held fixed so that the search
        # budget stays identical to the random-search baseline.
        return search_grid(self.tune_depth)

    def evaluation_budget(self, params: Dict[str, Any]) -> Optional[int]:
        return ga_evaluation_budget(self.ga_config)

    def fit(self, X: np.ndarray, y: np.ndarray, params: Dict[str, Any], seed: int) -> FittedModel:
        context = _build_search_context(self.tree_config, self.fitness_config, X, y, params, seed)

        config = GAConfig(
            population_size=self.ga_config["population_size"],
            n_generations=self.ga_config["n_generations"],
            crossover_prob=self.ga_config["crossover_prob"],
            mutation_prob=self.ga_config["mutation_prob"],
            tournament_size=self.ga_config["tournament_size"],
            elitism_ratio=self.ga_config["elitism_ratio"],
            mutation_types=self.ga_config.get("mutation_types"),
            random_state=seed,
            early_stopping_rounds=self.ga_config.get("early_stopping_rounds"),
            early_stopping_tol=self.ga_config.get("early_stopping_tol", 1e-6),
        )
        engine = GAEngine(
            config=config,
            initializer=context.initializer,
            fitness_function=context.fitness,
            mutation=Mutation(
                n_features=X.shape[1],
                feature_ranges=_feature_ranges(context.X_train),
                # The GA-train split only: drawing thresholds from values in the
                # validation split would leak it into the search it is meant to
                # hold out.
                X=context.X_train,
                min_samples_leaf=self.tree_config["min_samples_leaf"],
                split_strategy=self.tree_config.get("split_strategy", MIDPOINT_STRATEGY),
            ),
            repair=repair_from_config(self.tree_config, context.X_train, context.y_train),
        )
        best = engine.evolve(
            context.X_train,
            context.y_train,
            verbose=False,
            X_val=context.X_val,
            y_val=context.y_val,
        )

        # Structure was selected on the validation split; refitting the leaves on
        # the whole fold's training data is the standard follow-up and costs
        # nothing in validity, since no structural choice is made here.
        predictor = TreePredictor()
        predictor.fit_leaf_predictions(best, X, y)
        measures = ga_tree_complexity(best, X)

        return FittedModel(
            predict_fn=lambda X_new: predictor.predict(best, X_new),
            n_nodes=measures["n_nodes"],
            n_leaves=measures["n_leaves"],
            max_depth=measures["max_depth"],
            n_features_used=measures["n_features_used"],
            mean_path_length=measures["mean_path_length"],
            n_evaluations=context.fitness.count,
        )


class RandomTreeSearch(BenchmarkMethod):
    """Budget-matched random search over the GA's own tree space.

    Draws trees from the same :class:`TreeInitializer`, scores them with the
    same :class:`FitnessCalculator`, and keeps the best. The only difference
    from the GA is that there is no selection, crossover or mutation — so any
    gap between the two is attributable to the evolutionary machinery and
    nothing else.

    This is the K1 kill criterion: if this matches the GA, there is no paper.
    """

    name = "Random Search"

    def __init__(
        self,
        ga_config: Dict[str, Any],
        tree_config: Dict[str, Any],
        fitness_config: Dict[str, Any],
        tune: bool = True,
        tune_depth: bool = True,
    ):
        self.ga_config = dict(ga_config)
        self.tree_config = dict(tree_config)
        self.fitness_config = dict(fitness_config)
        self.tune = tune
        self.tune_depth = tune_depth

    def param_grid(self, X: np.ndarray, y: np.ndarray) -> List[Dict[str, Any]]:
        if not self.tune:
            return [{}]
        # Deliberately the same grid as GATreeMethod: an advantage must not come
        # from one method being tuned over a richer space than the other.
        return search_grid(self.tune_depth)

    def evaluation_budget(self, params: Dict[str, Any]) -> Optional[int]:
        # Deliberately the GA's formula, not pop * gens: the two must spend the
        # same number of evaluations for K1 to mean anything.
        return ga_evaluation_budget(self.ga_config)

    def fit(self, X: np.ndarray, y: np.ndarray, params: Dict[str, Any], seed: int) -> FittedModel:
        import random as _random

        context = _build_search_context(self.tree_config, self.fitness_config, X, y, params, seed)

        _random.seed(seed)
        np.random.seed(seed)

        budget = self.evaluation_budget(params)
        best_tree = None
        best_fitness = -np.inf
        for _ in range(budget):
            candidate = context.initializer.create_random_tree(context.X_train, context.y_train)
            fitness = context.score(candidate)
            if fitness > best_fitness:
                best_fitness = fitness
                best_tree = candidate

        # Same refit-on-everything step the GA gets, for the same reason.
        predictor = TreePredictor()
        predictor.fit_leaf_predictions(best_tree, X, y)
        measures = ga_tree_complexity(best_tree, X)

        return FittedModel(
            predict_fn=lambda X_new: predictor.predict(best_tree, X_new),
            n_nodes=measures["n_nodes"],
            n_leaves=measures["n_leaves"],
            max_depth=measures["max_depth"],
            n_features_used=measures["n_features_used"],
            mean_path_length=measures["mean_path_length"],
            n_evaluations=context.fitness.count,
        )


class PrunedCARTMethod(BenchmarkMethod):
    """CART with cost-complexity pruning, ``ccp_alpha`` tuned by inner CV.

    This is the baseline the accuracy and size claims must beat. The previous
    protocol compared against a fixed ``max_depth=6`` tree with no pruning,
    which is not a tuned baseline and made the size comparison meaningless.
    """

    name = "CART (pruned)"

    def __init__(self, tree_config: Dict[str, Any], depth_grid=DEFAULT_DEPTH_GRID):
        self.tree_config = dict(tree_config)
        self.depth_grid = tuple(depth_grid)

    def param_grid(self, X: np.ndarray, y: np.ndarray) -> List[Dict[str, Any]]:
        # The pruning path is data-dependent, so derive it from the same data
        # the inner selection will score on.
        probe = DecisionTreeClassifier(
            random_state=0,
            min_samples_split=self.tree_config["min_samples_split"],
            min_samples_leaf=self.tree_config["min_samples_leaf"],
        )
        try:
            path = probe.cost_complexity_pruning_path(X, y)
            alphas = np.unique(path.ccp_alphas)
            alphas = alphas[alphas >= 0]
        except (ValueError, AttributeError):  # degenerate data
            alphas = np.array([0.0])

        if len(alphas) > MAX_CCP_ALPHAS:
            idx = np.linspace(0, len(alphas) - 1, MAX_CCP_ALPHAS).astype(int)
            alphas = alphas[idx]

        return [
            {"ccp_alpha": float(alpha), "max_depth": depth}
            for alpha in alphas
            for depth in self.depth_grid
        ]

    def fit(self, X: np.ndarray, y: np.ndarray, params: Dict[str, Any], seed: int) -> FittedModel:
        estimator = DecisionTreeClassifier(
            max_depth=params.get("max_depth"),
            ccp_alpha=params.get("ccp_alpha", 0.0),
            min_samples_split=self.tree_config["min_samples_split"],
            min_samples_leaf=self.tree_config["min_samples_leaf"],
            random_state=seed,
        )
        estimator.fit(X, y)
        measures = sklearn_tree_complexity(estimator, X)
        return FittedModel(predict_fn=estimator.predict, **measures)


class UnconstrainedCARTMethod(BenchmarkMethod):
    """CART grown to purity — the accuracy ceiling for a single tree.

    No depth cap, no pruning, no minimum-sample constraints beyond sklearn's
    defaults. It exists to show what the GA gives up, and nothing about it is
    tuned.
    """

    name = "CART (unconstrained)"

    def param_grid(self, X: np.ndarray, y: np.ndarray) -> List[Dict[str, Any]]:
        return [{}]

    def fit(self, X: np.ndarray, y: np.ndarray, params: Dict[str, Any], seed: int) -> FittedModel:
        estimator = DecisionTreeClassifier(max_depth=None, random_state=seed)
        estimator.fit(X, y)
        measures = sklearn_tree_complexity(estimator, X)
        return FittedModel(predict_fn=estimator.predict, **measures)


class RandomForestMethod(BenchmarkMethod):
    """Random forest — an accuracy reference, not an interpretability one."""

    name = "Random Forest"

    def __init__(self, n_estimators: int = 100):
        self.n_estimators = n_estimators

    def param_grid(self, X: np.ndarray, y: np.ndarray) -> List[Dict[str, Any]]:
        return [{"max_depth": depth} for depth in (None, 6, 10)]

    def fit(self, X: np.ndarray, y: np.ndarray, params: Dict[str, Any], seed: int) -> FittedModel:
        estimator = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=params.get("max_depth"),
            random_state=seed,
            n_jobs=1,
        )
        estimator.fit(X, y)

        per_tree = [sklearn_tree_complexity(t, X) for t in estimator.estimators_]
        return FittedModel(
            predict_fn=estimator.predict,
            n_nodes=int(sum(m["n_nodes"] for m in per_tree)),
            n_leaves=int(sum(m["n_leaves"] for m in per_tree)),
            max_depth=int(max(m["max_depth"] for m in per_tree)),
            n_features_used=X.shape[1],
            mean_path_length=float(sum(m["mean_path_length"] for m in per_tree)),
        )
