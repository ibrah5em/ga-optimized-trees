"""Unit tests for frontier dominance and hypervolume.

The load-bearing test here is that hypervolume is computed over *distinct
objective vectors*: a front of many structurally different trees sitting on a
handful of objective points is worth the area of those points and no more.
Conflating front size with frontier quality overstates the result ~9x on real
data, which is the failure this module exists to prevent.
"""

import numpy as np
import pytest
from sklearn.datasets import load_iris

from ga_trees.evaluation.hypervolume import (
    Frontier,
    cart_pruning_frontier,
    compare_frontiers,
    distinct_objective_vectors,
    frontier,
    frontier_from_trees,
    hypervolume,
    non_dominated,
    reference_nodes_for,
)
from ga_trees.genotype.tree_genotype import TreeGenotype, create_internal_node, create_leaf_node


def _tree(n_leaves: int, accuracy: float) -> TreeGenotype:
    """A left-spine tree with *n_leaves* leaves, tagged with an accuracy."""
    node = create_leaf_node(0, depth=n_leaves - 1)
    for depth in range(n_leaves - 2, -1, -1):
        node = create_internal_node(0, 0.5, node, create_leaf_node(1, depth + 1), depth)
    tree = TreeGenotype(root=node, n_features=2, n_classes=2, max_depth=n_leaves)
    tree.accuracy_ = accuracy
    return tree


class TestDistinctObjectiveVectors:
    def test_duplicates_collapse(self):
        points = [(0.9, 5), (0.9, 5), (0.8, 3), (0.9, 5)]
        assert distinct_objective_vectors(points).shape == (2, 2)

    def test_first_seen_order_is_kept(self):
        result = distinct_objective_vectors([(0.9, 5), (0.8, 3), (0.9, 5)])
        assert result[0].tolist() == [0.9, 5.0]

    def test_float_noise_in_the_last_ulp_still_collapses(self):
        jittered = 0.9 + 1e-15
        assert distinct_objective_vectors([(0.9, 5), (jittered, 5)]).shape == (1, 2)

    def test_empty_input(self):
        assert distinct_objective_vectors([]).shape == (0, 2)

    def test_wrong_shape_rejected(self):
        with pytest.raises(ValueError, match=r"\(n, 2\)"):
            distinct_objective_vectors([(0.9, 5, 1)])


class TestNonDominated:
    def test_dominated_point_removed(self):
        # (0.8, 10) is beaten by (0.9, 5) on both axes.
        result = non_dominated(np.array([[0.9, 5.0], [0.8, 10.0]]))
        assert result.tolist() == [[0.9, 5.0]]

    def test_genuine_tradeoff_is_kept(self):
        points = np.array([[0.80, 3.0], [0.90, 9.0], [0.85, 5.0]])
        assert non_dominated(points).shape[0] == 3

    def test_output_is_sorted_by_node_count(self):
        points = np.array([[0.90, 9.0], [0.80, 3.0], [0.85, 5.0]])
        assert non_dominated(points)[:, 1].tolist() == [3.0, 5.0, 9.0]

    def test_empty_input(self):
        assert non_dominated(np.empty((0, 2))).shape[0] == 0


class TestFrontier:
    def test_counts_distinguish_size_from_distinct_points(self):
        # 27 trees, 3 distinct objective points — the iris case.
        points = [(0.90, 5), (0.95, 9), (0.85, 3)] * 9
        front = frontier(points)
        assert front.n_evaluated == 27
        assert front.n_distinct == 3
        assert len(front) == 3

    def test_dominated_points_do_not_count_toward_the_frontier(self):
        front = frontier([(0.9, 5), (0.8, 9), (0.7, 12)])
        assert front.n_distinct == 3
        assert len(front) == 1

    def test_from_trees_uses_real_node_counts(self):
        front = frontier_from_trees([_tree(2, 0.8), _tree(4, 0.9)])
        assert front.node_counts.tolist() == [3.0, 7.0]

    def test_from_trees_prefers_supplied_accuracies(self):
        front = frontier_from_trees([_tree(2, 0.8)], accuracies=[0.5])
        assert front.accuracies.tolist() == [0.5]

    def test_from_trees_rejects_mismatched_accuracy_count(self):
        with pytest.raises(ValueError, match="2 accuracies for 1 trees"):
            frontier_from_trees([_tree(2, 0.8)], accuracies=[0.5, 0.6])

    def test_unevaluated_tree_is_refused(self):
        tree = _tree(2, 0.8)
        tree.accuracy_ = None
        with pytest.raises(ValueError, match="no accuracy_"):
            frontier_from_trees([tree])


class TestHypervolume:
    def test_single_point_is_a_rectangle(self):
        # (0.9, 5) against reference (0, 20) covers 0.9 x 15.
        front = frontier([(0.9, 5.0)])
        assert hypervolume(front, reference_nodes=20.0) == pytest.approx(0.9 * 15.0)

    def test_two_points_sweep_without_double_counting(self):
        # (0.8, 3): 0.8 x 17 = 13.6; (0.9, 5) adds (0.9-0.8) x 15 = 1.5
        front = frontier([(0.8, 3.0), (0.9, 5.0)])
        assert hypervolume(front, reference_nodes=20.0) == pytest.approx(13.6 + 1.5)

    def test_duplicate_points_do_not_inflate_the_volume(self):
        one = hypervolume(frontier([(0.9, 5.0)]), reference_nodes=20.0)
        many = hypervolume(frontier([(0.9, 5.0)] * 50), reference_nodes=20.0)
        assert one == pytest.approx(many)

    def test_dominating_frontier_scores_higher(self):
        better = frontier([(0.95, 4.0), (0.85, 2.0)])
        worse = frontier([(0.90, 6.0), (0.80, 3.0)])
        assert hypervolume(better, 20.0) > hypervolume(worse, 20.0)

    def test_empty_frontier_is_zero(self):
        assert hypervolume(frontier([]), reference_nodes=20.0) == 0.0

    def test_points_outside_the_reference_box_contribute_nothing(self):
        front = Frontier(points=np.array([[0.9, 50.0]]), n_evaluated=1, n_distinct=1)
        assert hypervolume(front, reference_nodes=20.0) == 0.0

    def test_more_nodes_at_equal_accuracy_is_worth_less(self):
        compact = hypervolume(frontier([(0.9, 3.0)]), 20.0)
        bloated = hypervolume(frontier([(0.9, 15.0)]), 20.0)
        assert compact > bloated


class TestSharedReferencePoint:
    def test_reference_covers_every_method(self):
        frontiers = {"a": frontier([(0.9, 5.0)]), "b": frontier([(0.8, 40.0)])}
        assert reference_nodes_for(frontiers) == pytest.approx(41.0)

    def test_all_methods_scored_against_the_same_box(self):
        frontiers = {"ga": frontier([(0.90, 6.0)]), "cart": frontier([(0.92, 30.0)])}
        volumes = compare_frontiers(frontiers)
        reference = reference_nodes_for(frontiers)
        assert volumes["ga"] == pytest.approx(0.90 * (reference - 6.0))
        assert volumes["cart"] == pytest.approx(0.92 * (reference - 30.0))

    def test_empty_frontiers_cannot_produce_a_reference(self):
        with pytest.raises(ValueError, match="empty frontiers"):
            reference_nodes_for({"a": frontier([])})


class TestCartPruningFrontier:
    def test_traces_a_real_tradeoff_on_iris(self):
        X, y = load_iris(return_X_y=True)
        front, alphas = cart_pruning_frontier(X[:100], y[:100], X[100:], y[100:])
        assert len(alphas) >= 1
        assert len(front) >= 1
        # Non-dominated by construction: node counts ascend, accuracy with them.
        assert list(front.node_counts) == sorted(front.node_counts)
        assert list(front.accuracies) == sorted(front.accuracies)

    def test_frontier_is_comparable_to_a_ga_frontier(self):
        X, y = load_iris(return_X_y=True)
        cart, _ = cart_pruning_frontier(X[:100], y[:100], X[100:], y[100:])
        ga = frontier([(0.9, 3.0), (0.95, 7.0)])
        volumes = compare_frontiers({"GA": ga, "CART": cart})
        assert set(volumes) == {"GA", "CART"}
        assert all(v >= 0 for v in volumes.values())
