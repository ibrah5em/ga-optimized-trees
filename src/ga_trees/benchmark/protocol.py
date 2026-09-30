"""Common interface every benchmarked method implements.

The point of a shared interface is that ``nested_cv`` cannot treat one method
more favourably than another by accident. Every method gets the same inner-CV
tuning, the same seeds, and reports the same complexity measures.

Reported interpretability is fixed by the pre-registered kill criterion K4: number of
leaves, mean weighted decision-path length, and number of distinct features.
The composite interpretability score is a search heuristic and deliberately not
part of this interface.
"""

import abc
from typing import Any, Callable, Dict, List, Optional

import numpy as np


class FittedModel:
    """A trained model plus the complexity measures required for reporting.

    Parameters
    ----------
    predict_fn : callable
        ``predict_fn(X) -> array of predictions``.
    n_nodes : int
        Total nodes in the model. For ensembles, summed over estimators.
    n_leaves : int
        Leaf count — the primary complexity axis under K4.
    max_depth : int
        Deepest root-to-leaf path.
    n_features_used : int
        Distinct features appearing in any split.
    mean_path_length : float
        Sample-weighted mean decision-path length: the average number of tests a
        prediction actually costs. Less gameable than depth, which is set by the
        single longest branch.
    n_evaluations : int
        Candidate models evaluated to produce this fit. Used to verify budget
        matching rather than assume it; 0 for methods that do not search.
    """

    def __init__(
        self,
        predict_fn: Callable[[np.ndarray], np.ndarray],
        n_nodes: int,
        n_leaves: int,
        max_depth: int,
        n_features_used: int,
        mean_path_length: float,
        n_evaluations: int = 0,
    ):
        self._predict_fn = predict_fn
        self.n_nodes = int(n_nodes)
        self.n_leaves = int(n_leaves)
        self.max_depth = int(max_depth)
        self.n_features_used = int(n_features_used)
        self.mean_path_length = float(mean_path_length)
        self.n_evaluations = int(n_evaluations)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict labels for ``X``."""
        return self._predict_fn(X)

    def complexity(self) -> Dict[str, float]:
        """Complexity measures as a flat dict, for result rows."""
        return {
            "nodes": self.n_nodes,
            "leaves": self.n_leaves,
            "depth": self.max_depth,
            "features_used": self.n_features_used,
            "mean_path_length": self.mean_path_length,
            "evaluations": self.n_evaluations,
        }


class BenchmarkMethod(abc.ABC):
    """A method that can be tuned by inner CV and fitted on an outer fold.

    Subclasses must be stateless between ``fit`` calls: the harness reuses one
    instance across every fold, and any state carried over would leak training
    data between folds.
    """

    #: Human-readable name used in result rows and statistical comparisons.
    name = "unnamed"

    @abc.abstractmethod
    def param_grid(self, X: np.ndarray, y: np.ndarray) -> List[Dict[str, Any]]:
        """Candidate hyperparameter settings for inner-CV selection.

        Receives the *inner training* data so that data-dependent grids (a
        cost-complexity pruning path, for instance) can be derived honestly
        from the same data the selection sees.

        Returns
        -------
        list of dict
            Never empty. A method with nothing to tune returns ``[{}]``.
        """

    @abc.abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray, params: Dict[str, Any], seed: int) -> FittedModel:
        """Fit on ``(X, y)`` with ``params`` and return a :class:`FittedModel`."""

    def evaluation_budget(self, params: Dict[str, Any]) -> Optional[int]:
        """Candidate models this method may evaluate under ``params``.

        Returns ``None`` when the method does not search a space, which is what
        makes budget matching meaningful only between the methods that do.
        """
        return None


def sklearn_tree_complexity(estimator, X: np.ndarray) -> Dict[str, float]:
    """Complexity measures for a fitted sklearn decision tree.

    Parameters
    ----------
    estimator : DecisionTreeClassifier
        A fitted tree.
    X : ndarray
        Data used to weight the mean path length — normally the training set.

    Returns
    -------
    dict
        ``n_nodes``, ``n_leaves``, ``max_depth``, ``n_features_used``,
        ``mean_path_length``.
    """
    tree = estimator.tree_
    is_leaf = tree.children_left == -1
    n_leaves = int(np.sum(is_leaf))

    # decision_path marks every node on a sample's route, root included, so the
    # number of *tests* is one less than the number of nodes visited.
    paths = estimator.decision_path(X)
    mean_path_length = float(np.mean(np.asarray(paths.sum(axis=1)).ravel() - 1))

    used = {int(f) for f in tree.feature if f >= 0}

    return {
        "n_nodes": int(tree.node_count),
        "n_leaves": n_leaves,
        "max_depth": int(estimator.get_depth()),
        "n_features_used": len(used),
        "mean_path_length": mean_path_length,
    }


def ga_tree_complexity(tree, X: np.ndarray) -> Dict[str, float]:
    """Complexity measures for a fitted :class:`TreeGenotype`.

    Path length is measured by routing every sample through the tree, matching
    what :func:`sklearn_tree_complexity` reports so the two are comparable.
    """
    depths = []
    for row in X:
        node = tree.root
        depth = 0
        while node is not None and not node.is_leaf():
            if row[node.feature_idx] <= node.threshold:
                nxt = node.left_child
            else:
                nxt = node.right_child
            if nxt is None:
                break
            node = nxt
            depth += 1
        depths.append(depth)

    return {
        "n_nodes": tree.get_num_nodes(),
        "n_leaves": tree.get_num_leaves(),
        "max_depth": tree.get_depth(),
        "n_features_used": tree.get_num_features_used(),
        "mean_path_length": float(np.mean(depths)) if depths else 0.0,
    }
