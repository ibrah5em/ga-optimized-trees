"""Frontier quality: dominance filtering and 2-D hypervolume.

This is the machinery behind H1 in ``paper/PREREGISTRATION.md`` — whether the
evolved accuracy--complexity frontier dominates the one obtainable from CART's
cost-complexity pruning path.

Two things are fixed by the pre-registration and enforced here rather than left
to the caller:

* **The complexity axis is node count**, not the composite interpretability
  score. The composite score is a search heuristic (K4) and may not appear as a
  reported outcome.
* **The reference point is (accuracy = 0, nodes = max over all methods on that
  dataset)**, so every method on a dataset is measured against the same box.
  Hypervolumes computed against different reference points are not comparable,
  which is the most common way this metric is misreported.

And one thing is enforced because getting it wrong inflates the result by
roughly an order of magnitude: **hypervolume is computed over distinct objective
vectors**. A Pareto front of 27 structurally distinct trees occupying 3 distinct
objective points is a 3-point frontier. Reporting front size in its place
overstates the frontier ~9x — measured on iris after the duplicate-elimination
fix. :func:`frontier` therefore deduplicates before anything else, and
:class:`Frontier` reports both counts so the difference stays visible.
"""

import logging
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)

#: Decimal places objective values are rounded to before deduplication. Float
#: arithmetic in crossover and mutation makes otherwise identical trees differ
#: in the last ulp; accuracy is a ratio of counts, so 9 places is far finer than
#: any real difference.
OBJECTIVE_DECIMALS = 9


@dataclass
class Frontier:
    """A set of non-dominated (accuracy, nodes) points.

    Attributes
    ----------
    points : ndarray
        Shape ``(n, 2)``: accuracy (maximised) and node count (minimised),
        sorted by node count ascending.
    n_evaluated : int
        Candidates supplied before deduplication and dominance filtering.
    n_distinct : int
        Distinct objective vectors among them. The honest measure of how much
        of the frontier a run actually found.
    """

    points: np.ndarray
    n_evaluated: int
    n_distinct: int

    def __len__(self) -> int:
        return int(self.points.shape[0])

    @property
    def accuracies(self) -> np.ndarray:
        return self.points[:, 0]

    @property
    def node_counts(self) -> np.ndarray:
        return self.points[:, 1]


def distinct_objective_vectors(
    points: Iterable[Sequence[float]], decimals: int = OBJECTIVE_DECIMALS
) -> np.ndarray:
    """Unique (accuracy, nodes) pairs, in first-seen order."""
    array = np.asarray(list(points), dtype=float)
    if array.size == 0:
        return np.empty((0, 2), dtype=float)
    if array.ndim != 2 or array.shape[1] != 2:
        raise ValueError(f"Expected an (n, 2) array of objective vectors, got {array.shape}.")

    rounded = np.round(array, decimals)
    _, first_index = np.unique(rounded, axis=0, return_index=True)
    return array[np.sort(first_index)]


def non_dominated(points: np.ndarray) -> np.ndarray:
    """Keep points no other point beats on both objectives.

    A point is dominated when another has accuracy at least as high *and* node
    count no larger, with at least one strict. Ties on both objectives are
    already gone by the time this runs.
    """
    if points.shape[0] == 0:
        return points

    accuracy = points[:, 0]
    nodes = points[:, 1]
    keep = np.ones(points.shape[0], dtype=bool)

    for index in range(points.shape[0]):
        better_or_equal = (accuracy >= accuracy[index]) & (nodes <= nodes[index])
        strictly_better = (accuracy > accuracy[index]) | (nodes < nodes[index])
        if np.any(better_or_equal & strictly_better):
            keep[index] = False

    survivors = points[keep]
    return survivors[np.argsort(survivors[:, 1], kind="stable")]


def frontier(points: Iterable[Sequence[float]]) -> Frontier:
    """Build a :class:`Frontier` from raw (accuracy, nodes) pairs."""
    array = np.asarray(list(points), dtype=float)
    n_evaluated = int(array.shape[0]) if array.size else 0

    distinct = distinct_objective_vectors(array) if n_evaluated else np.empty((0, 2))
    return Frontier(
        points=non_dominated(distinct),
        n_evaluated=n_evaluated,
        n_distinct=int(distinct.shape[0]),
    )


def frontier_from_trees(trees: Iterable, accuracies: Optional[Sequence[float]] = None) -> Frontier:
    """Build a frontier from fitted :class:`TreeGenotype` objects.

    Parameters
    ----------
    trees : iterable
        Trees carrying ``accuracy_`` and answering ``get_num_nodes()``.
    accuracies : sequence of float, optional
        Held-out accuracies, one per tree, overriding ``tree.accuracy_``. Supply
        these for anything reportable: ``accuracy_`` is whatever the search
        scored the tree on, which is training or GA-validation data, not the
        outer test fold.
    """
    trees = list(trees)
    if accuracies is not None and len(accuracies) != len(trees):
        raise ValueError(f"Got {len(accuracies)} accuracies for {len(trees)} trees.")

    pairs = []
    for index, tree in enumerate(trees):
        accuracy = accuracies[index] if accuracies is not None else tree.accuracy_
        if accuracy is None:
            raise ValueError(
                "Tree has no accuracy_; evaluate the population before building a frontier."
            )
        pairs.append((float(accuracy), float(tree.get_num_nodes())))
    return frontier(pairs)


def hypervolume(front: Frontier, reference_nodes: float, reference_accuracy: float = 0.0) -> float:
    """Area dominated by *front*, bounded by the reference point.

    Each point ``(a, n)`` claims the rectangle
    ``[reference_accuracy, a] x [n, reference_nodes]``; the hypervolume is the
    area of their union, computed exactly by a sweep rather than sampled.

    Parameters
    ----------
    front : Frontier
        Non-dominated points, node count ascending.
    reference_nodes : float
        Node count of the worst acceptable model — per the pre-registration, the
        maximum over *all* methods on that dataset. Must exceed every node count
        in *front*, or the excess points contribute nothing.
    reference_accuracy : float
        Accuracy floor, 0.0 by the pre-registration.

    Returns
    -------
    float
        Dominated area. 0.0 for an empty frontier.
    """
    points = front.points
    if points.shape[0] == 0:
        return 0.0

    usable = points[(points[:, 1] <= reference_nodes) & (points[:, 0] >= reference_accuracy)]
    if usable.shape[0] == 0:
        logger.warning(
            "No frontier point is inside the reference box (nodes <= %.1f, accuracy >= %.3f).",
            reference_nodes,
            reference_accuracy,
        )
        return 0.0

    # Node count ascending; on a true front accuracy then ascends with it, so
    # each successive point contributes a slab of width (a_i - a_{i-1}).
    usable = usable[np.argsort(usable[:, 1], kind="stable")]
    area = 0.0
    previous_accuracy = reference_accuracy
    for accuracy, nodes in usable:
        width = accuracy - previous_accuracy
        if width <= 0:
            continue
        area += width * (reference_nodes - nodes)
        previous_accuracy = accuracy
    return float(area)


def reference_nodes_for(frontiers: Dict[str, Frontier], margin: float = 1.0) -> float:
    """Shared reference node count for every method on one dataset.

    Taken as the largest node count anywhere on the dataset, plus a margin so
    that the worst point still contributes non-zero area instead of collapsing
    the comparison to a tie at the boundary.
    """
    maxima = [float(front.node_counts.max()) for front in frontiers.values() if len(front) > 0]
    if not maxima:
        raise ValueError("Cannot derive a reference point from empty frontiers.")
    return max(maxima) + margin


def compare_frontiers(frontiers: Dict[str, Frontier], margin: float = 1.0) -> Dict[str, float]:
    """Hypervolume per method against one shared reference point.

    Parameters
    ----------
    frontiers : dict of str to Frontier
        One frontier per method, all on the same dataset.
    margin : float
        Added to the largest observed node count to form the reference.

    Returns
    -------
    dict
        Method name to hypervolume, comparable within this dataset only.
    """
    reference = reference_nodes_for(frontiers, margin=margin)
    return {name: hypervolume(front, reference) for name, front in frontiers.items()}


def cart_pruning_frontier(
    X: np.ndarray,
    y: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    max_alphas: int = 12,
    random_state: int = 0,
    **tree_kwargs,
) -> Tuple[Frontier, List[float]]:
    """The frontier CART's cost-complexity pruning path traces.

    This is H1's comparator: sweeping ``ccp_alpha`` is the standard way to get a
    range of tree sizes out of CART, and it is what the GA's single-run frontier
    has to beat to make the claim.

    Returns
    -------
    tuple
        The frontier, and the ``ccp_alpha`` values behind its points.
    """
    from sklearn.tree import DecisionTreeClassifier

    probe = DecisionTreeClassifier(random_state=random_state, **tree_kwargs)
    try:
        alphas = np.unique(probe.cost_complexity_pruning_path(X, y).ccp_alphas)
        alphas = alphas[alphas >= 0]
    except (ValueError, AttributeError):
        alphas = np.array([0.0])

    if len(alphas) > max_alphas:
        alphas = alphas[np.linspace(0, len(alphas) - 1, max_alphas).astype(int)]

    pairs = []
    for alpha in alphas:
        estimator = DecisionTreeClassifier(
            ccp_alpha=float(alpha), random_state=random_state, **tree_kwargs
        )
        estimator.fit(X, y)
        accuracy = float(np.mean(estimator.predict(X_test) == y_test))
        pairs.append((accuracy, float(estimator.tree_.node_count)))

    return frontier(pairs), [float(a) for a in alphas]
