"""GOSDT frontier adapter (Phase 4, exploratory)."""

import numpy as np
import pytest

from ga_trees.benchmark.gosdt_frontier import _count_nodes


def test_node_count_of_a_stump_and_a_split():
    assert _count_nodes({"prediction": 1}) == 1
    split = {"feature": 0, "true": {"prediction": 0}, "false": {"prediction": 1}}
    assert _count_nodes(split) == 3
    assert _count_nodes({"feature": 1, "true": split, "false": {"prediction": 0}}) == 5


def test_path_scores_like_a_sklearn_tree():
    pytest.importorskip("gosdt")
    from ga_trees.benchmark.frontiers import _score_candidates
    from ga_trees.benchmark.gosdt_frontier import GOSDTPathFrontier

    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 3))
    y = (X[:, 0] > 0.2).astype(int)
    method = GOSDTPathFrontier(max_depth=3, regularizations=(0.01, 0.2), time_limit=10)
    models, evaluations = method.build(X[:150], y[:150], seed=0)
    assert evaluations == 0 and len(models) == 2
    points = _score_candidates(models, X[:150], y[:150], X[150:], y[150:])
    accuracies = [a for a, _ in points]
    nodes = [n for _, n in points]
    assert max(accuracies) > 0.85
    assert min(nodes) >= 1 and all(n % 2 == 1 for n in nodes)
