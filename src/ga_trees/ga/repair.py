"""Data-aware constraint repair.

``min_samples_split`` and ``min_samples_leaf`` are enforced by
:class:`~ga_trees.ga.engine.TreeInitializer` and nowhere else. Crossover grafts a
subtree built for one region of the data into another, and ``expand_leaf`` /
``feature_replacement`` choose splits without re-checking how many samples end
up on each side, so evolved trees routinely contain leaves that no training
sample — or only one or two — ever reaches. Without repair, the configured
constraints describe the initial population, not the evolved trees.

:func:`repair_constraints` routes the training data through a tree and collapses,
top-down, every split that the constraints would not have allowed CART to make:

* an internal node reached by fewer than ``min_samples_split`` samples, or
* a split that sends fewer than ``min_samples_leaf`` samples to either side.

The collapsed node becomes a leaf predicting the majority class of the samples
that reach it. Collapsing only ever removes nodes, so repair cannot grow a tree
and cannot break ``max_depth``.

Repair is a correctness fix, not a performance lever, and it is *off* by
default (``tree.repair_constraints``) so that ``configs/paper.yaml`` still
reproduces the original benchmark run bit for bit.
"""

from typing import Optional

import numpy as np

from ga_trees.genotype.tree_genotype import Node, TreeGenotype


def _majority(y: np.ndarray, fallback):
    if y.size == 0:
        return fallback
    values, counts = np.unique(y, return_counts=True)
    return int(values[np.argmax(counts)])


def _leftmost_prediction(node: Node):
    while node is not None and not node.is_leaf():
        node = node.left_child
    if node is None or node.prediction is None:
        return 0
    return node.prediction


def _collapse(node: Node, prediction) -> None:
    node.node_type = "leaf"
    node.prediction = prediction
    node.left_child = None
    node.right_child = None
    node.feature_idx = None
    node.threshold = None


def _violates(node: Node, X: np.ndarray, indices: np.ndarray, split: int, leaf: int) -> bool:
    if indices.size < split:
        return True
    goes_left = X[indices, node.feature_idx] <= node.threshold
    n_left = int(goes_left.sum())
    return n_left < leaf or indices.size - n_left < leaf


def count_violations(
    tree: TreeGenotype,
    X: np.ndarray,
    min_samples_split: Optional[int] = None,
    min_samples_leaf: Optional[int] = None,
) -> int:
    """Number of internal nodes whose split the constraints would forbid.

    Counted on the tree as it stands, without collapsing anything, so a
    violation beneath another violation is counted too.
    """
    split = tree.min_samples_split if min_samples_split is None else min_samples_split
    leaf = tree.min_samples_leaf if min_samples_leaf is None else min_samples_leaf
    violations = 0
    stack = [(tree.root, np.arange(X.shape[0]))]
    while stack:
        node, indices = stack.pop()
        if node is None or node.is_leaf():
            continue
        if _violates(node, X, indices, split, leaf):
            violations += 1
        goes_left = X[indices, node.feature_idx] <= node.threshold
        stack.append((node.left_child, indices[goes_left]))
        stack.append((node.right_child, indices[~goes_left]))
    return violations


def repair_constraints(
    tree: TreeGenotype,
    X: np.ndarray,
    y: Optional[np.ndarray] = None,
    min_samples_split: Optional[int] = None,
    min_samples_leaf: Optional[int] = None,
) -> TreeGenotype:
    """Collapse every split the sample-count constraints forbid. Mutates *tree*.

    Args:
        tree: Tree to repair in place.
        X: Data to route — the GA-training split, the same rows leaf
            predictions are fitted on. Never the validation split.
        y: Labels for *X*, used to set the prediction of a collapsed node. When
            omitted the collapsed node inherits its leftmost leaf's prediction,
            as ``prune_subtree`` does; fitness evaluation refits leaves anyway.
        min_samples_split: Defaults to ``tree.min_samples_split``.
        min_samples_leaf: Defaults to ``tree.min_samples_leaf``.

    Returns:
        The same *tree*, for chaining.
    """
    split = tree.min_samples_split if min_samples_split is None else min_samples_split
    leaf = tree.min_samples_leaf if min_samples_leaf is None else min_samples_leaf

    stack = [(tree.root, np.arange(X.shape[0]))]
    while stack:
        node, indices = stack.pop()
        if node is None or node.is_leaf():
            continue
        if _violates(node, X, indices, split, leaf):
            fallback = _leftmost_prediction(node)
            _collapse(node, _majority(y[indices], fallback) if y is not None else fallback)
            continue
        goes_left = X[indices, node.feature_idx] <= node.threshold
        stack.append((node.left_child, indices[goes_left]))
        stack.append((node.right_child, indices[~goes_left]))
    return tree


def repair_from_config(tree_config: dict, X: np.ndarray, y: np.ndarray):
    """The repair callable a ``tree`` config section asks for, or ``None``.

    Reads ``repair_constraints`` (default False) and binds the constraints and
    the GA-training data, so engines only ever see ``(tree) -> tree``.
    """
    if not tree_config.get("repair_constraints", False):
        return None
    split = int(tree_config["min_samples_split"])
    leaf = int(tree_config["min_samples_leaf"])

    def repair(tree: TreeGenotype) -> TreeGenotype:
        return repair_constraints(tree, X, y, min_samples_split=split, min_samples_leaf=leaf)

    return repair
