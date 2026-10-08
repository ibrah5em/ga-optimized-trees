"""Where split thresholds come from.

CART enumerates every midpoint between consecutive distinct observed values of a
feature and keeps the best one. This package originally drew thresholds
from ``uniform(feature_min, feature_max)``, which is a different and much weaker
distribution:

* Most of the interval between the minimum and the maximum contains no data. A
  threshold landing there produces a split that is either degenerate or
  indistinguishable from a neighbouring one, so the fitness evaluation it costs
  buys nothing.
* The draw ignores the shape of the feature. One outlier stretches the range and
  pushes almost every draw into empty space, and skewed features are hit hardest.
* Two thresholds that partition the samples identically are the same tree, so the
  effective search space is far smaller than the sampling suggests while the
  *cost* of exploring it is not.

That last point is why this matters when the GA is compared against random
search. Both drew from the same threshold distribution, so a handicap here suppresses both
equally and makes them look alike. Sampling from observed midpoints gives both
methods the same candidate set CART chooses from, which is the honest version of
the comparison.

The ``uniform`` strategy is retained so the split-point ablation
can isolate this change rather than assert it.
"""

import random
from typing import Optional

import numpy as np

#: Threshold sampling strategies.
MIDPOINT_STRATEGY = "midpoint"
UNIFORM_STRATEGY = "uniform"
VALID_SPLIT_STRATEGIES = frozenset({MIDPOINT_STRATEGY, UNIFORM_STRATEGY})

_EMPTY = np.empty(0, dtype=float)


def validate_split_strategy(strategy: str) -> str:
    """Return *strategy* if recognised, else raise.

    Args:
        strategy: One of ``"midpoint"`` or ``"uniform"``.

    Returns:
        The validated strategy string.

    Raises:
        ValueError: If *strategy* is not a known strategy.
    """
    if strategy not in VALID_SPLIT_STRATEGIES:
        raise ValueError(
            f"split_strategy must be one of {sorted(VALID_SPLIT_STRATEGIES)}, got '{strategy}'."
        )
    return strategy


def candidate_thresholds(values: np.ndarray, min_samples_leaf: int = 1) -> np.ndarray:
    """Split points that separate the observed values of one feature.

    The candidate set is the midpoints between consecutive distinct values —
    CART's candidate set — filtered to those leaving at least *min_samples_leaf*
    samples on each side. Filtering here rather than after the fact means a
    sampled threshold never has to be rejected, so a node is only turned into a
    leaf when the feature genuinely cannot be split.

    Args:
        values: Feature column of the samples reaching a node, shape
            ``(n_samples,)``.
        min_samples_leaf: Minimum samples each side of the split must retain.

    Returns:
        Sorted array of valid thresholds, empty if the feature cannot be split
        under the constraint.
    """
    values = np.asarray(values, dtype=float).ravel()
    n_samples = values.size
    if n_samples < 2:
        return _EMPTY

    unique, counts = np.unique(values, return_counts=True)
    if unique.size < 2:
        return _EMPTY

    midpoints = (unique[:-1] + unique[1:]) / 2.0

    # When two consecutive values are adjacent floats their midpoint rounds up to
    # the upper one, which would send that value's samples down the left branch
    # and break the sample counts computed below. Falling back to the lower value
    # keeps the partition — and therefore the counts — exactly as intended.
    # sklearn's splitter applies the same correction.
    collapsed = midpoints >= unique[1:]
    if collapsed.any():
        midpoints[collapsed] = unique[:-1][collapsed]

    if min_samples_leaf <= 1:
        return midpoints

    # cumsum(counts)[i] is the number of samples <= unique[i], which is exactly
    # the left-branch size for the threshold sitting between unique[i] and
    # unique[i+1].
    left_counts = np.cumsum(counts)[:-1]
    right_counts = n_samples - left_counts
    keep = (left_counts >= min_samples_leaf) & (right_counts >= min_samples_leaf)
    return midpoints[keep]


def sample_threshold(
    values: np.ndarray,
    min_samples_leaf: int = 1,
    strategy: str = MIDPOINT_STRATEGY,
) -> Optional[float]:
    """Draw one split threshold for a feature.

    Args:
        values: Feature column of the samples reaching a node.
        min_samples_leaf: Minimum samples each side of the split must retain.
            Ignored under the ``uniform`` strategy, which cannot honour it.
        strategy: ``"midpoint"`` to draw uniformly from the observed candidate
            midpoints, ``"uniform"`` for a uniform draw across the
            observed range.

    Returns:
        A threshold, or ``None`` when the feature admits no valid split.
    """
    if strategy == UNIFORM_STRATEGY:
        values = np.asarray(values, dtype=float).ravel()
        if values.size == 0:
            return None
        low, high = float(values.min()), float(values.max())
        if low == high:
            return None
        return random.uniform(low, high)

    candidates = candidate_thresholds(values, min_samples_leaf)
    if candidates.size == 0:
        return None
    return float(candidates[random.randrange(candidates.size)])


def samples_reaching(root, target, X: np.ndarray) -> Optional[np.ndarray]:
    """Row indices of *X* that reach *target*, or ``None`` if it is unreachable.

    Nodes are matched by object identity rather than ``node_id``: crossover
    copies whole subtrees without renumbering, so IDs are not unique within a
    tree once the population has been through a generation.

    Args:
        root: Root :class:`~ga_trees.genotype.tree_genotype.Node` to route from.
        target: The node to route to.
        X: Design matrix of the data being routed.

    Returns:
        Index array, or ``None`` if *target* is not in the tree below *root*.
    """
    stack = [(root, np.arange(X.shape[0]))]
    while stack:
        node, indices = stack.pop()
        if node is None:
            continue
        if node is target:
            return indices
        if node.is_leaf() or node.left_child is None or node.right_child is None:
            continue
        if node.feature_idx is None or node.threshold is None:
            continue
        goes_left = X[indices, node.feature_idx] <= node.threshold
        stack.append((node.left_child, indices[goes_left]))
        stack.append((node.right_child, indices[~goes_left]))
    return None


def step_threshold(
    current: float,
    candidates: np.ndarray,
    scale: float = 0.1,
) -> float:
    """Perturb *current* and snap the result onto an observed candidate.

    Gaussian jitter on its own usually leaves the threshold inside the same gap
    between observed values, where the partition — and so the fitness — is
    unchanged. The mutation still costs a fitness evaluation, which is the
    currency the GA vs random search budget match is denominated in. Snapping onto a candidate
    makes the step real, and the one-index nudge guarantees it is never a no-op.

    Args:
        current: The node's existing threshold.
        candidates: Valid thresholds at that node, sorted ascending, non-empty.
        scale: Jitter standard deviation as a fraction of the candidate spread.

    Returns:
        A threshold drawn from *candidates*, different from *current* whenever
        *candidates* holds more than one value.
    """
    spread = float(candidates[-1] - candidates[0])
    std = max(spread * scale, 1e-12)
    jittered = current + random.gauss(0, std)

    index = int(np.argmin(np.abs(candidates - jittered)))
    if float(candidates[index]) == current and candidates.size > 1:
        direction = 1 if jittered > current else -1
        index = int(np.clip(index + direction, 0, candidates.size - 1))
        if float(candidates[index]) == current:
            index = int(np.clip(index - direction, 0, candidates.size - 1))

    return float(candidates[index])
