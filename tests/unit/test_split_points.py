"""Unit tests for data-driven split points (Phase 2 item 2).

Covers:
- candidate_thresholds: midpoint enumeration and the min_samples_leaf filter
- sample_threshold: both strategies, and the unsplittable case
- samples_reaching: routing by identity, not by node_id
- step_threshold: perturbation lands on a candidate and is not a no-op
- TreeInitializer: every threshold sits between two observed values
- Mutation: thresholds are drawn from the data reaching the node
"""

import random

import numpy as np
import pytest
from sklearn.datasets import load_iris

from ga_trees.ga.engine import Mutation, TreeInitializer
from ga_trees.ga.split_points import (
    MIDPOINT_STRATEGY,
    UNIFORM_STRATEGY,
    candidate_thresholds,
    sample_threshold,
    samples_reaching,
    step_threshold,
    validate_split_strategy,
)
from ga_trees.genotype.tree_genotype import create_internal_node, create_leaf_node

# [1, 1, 2, 3, 3, 3, 5] -> unique [1, 2, 3, 5], counts [2, 1, 3, 1]
SKEWED = np.array([1.0, 1.0, 2.0, 3.0, 3.0, 3.0, 5.0])


class TestCandidateThresholds:
    def test_midpoints_between_consecutive_distinct_values(self):
        assert candidate_thresholds(SKEWED) == pytest.approx([1.5, 2.5, 4.0])

    def test_min_samples_leaf_filters_lopsided_splits(self):
        # left counts are [2, 3, 6], right counts [5, 4, 1]
        assert candidate_thresholds(SKEWED, min_samples_leaf=2) == pytest.approx([1.5, 2.5])
        assert candidate_thresholds(SKEWED, min_samples_leaf=3) == pytest.approx([2.5])
        assert candidate_thresholds(SKEWED, min_samples_leaf=4).size == 0

    def test_constant_feature_has_no_candidates(self):
        assert candidate_thresholds(np.ones(10)).size == 0

    def test_single_sample_has_no_candidates(self):
        assert candidate_thresholds(np.array([3.0])).size == 0

    def test_empty_input_has_no_candidates(self):
        assert candidate_thresholds(np.array([])).size == 0

    def test_every_candidate_actually_separates_the_data(self):
        rng = np.random.RandomState(0)
        values = rng.rand(200).round(2)
        for threshold in candidate_thresholds(values, min_samples_leaf=5):
            left = int(np.sum(values <= threshold))
            assert 5 <= left <= len(values) - 5

    def test_adjacent_floats_do_not_break_the_count_guarantee(self):
        # The midpoint of two adjacent floats rounds up to the upper one, which
        # would move that value's samples left and violate min_samples_leaf.
        low = 1.0
        high = np.nextafter(low, 2.0)
        values = np.array([low] * 5 + [high] * 5)
        for threshold in candidate_thresholds(values, min_samples_leaf=5):
            assert int(np.sum(values <= threshold)) == 5


class TestSampleThreshold:
    def test_midpoint_strategy_returns_a_candidate(self):
        random.seed(0)
        candidates = candidate_thresholds(SKEWED, min_samples_leaf=2)
        for _ in range(20):
            assert sample_threshold(SKEWED, min_samples_leaf=2) in candidates

    def test_returns_none_when_unsplittable(self):
        assert sample_threshold(np.ones(10)) is None
        assert sample_threshold(SKEWED, min_samples_leaf=4) is None

    def test_uniform_strategy_stays_in_range_but_ignores_the_data(self):
        random.seed(0)
        draws = [sample_threshold(SKEWED, strategy=UNIFORM_STRATEGY) for _ in range(200)]
        assert all(1.0 <= d <= 5.0 for d in draws)
        # The point of the old strategy's weakness: most draws land in the gap
        # between 3 and 5 where no sample lives.
        assert any(3.0 < d < 5.0 for d in draws)

    def test_uniform_strategy_returns_none_on_constant_feature(self):
        assert sample_threshold(np.ones(10), strategy=UNIFORM_STRATEGY) is None

    def test_unknown_strategy_rejected(self):
        with pytest.raises(ValueError, match="split_strategy"):
            validate_split_strategy("gaussian")


class TestSamplesReaching:
    def _tree(self):
        left = create_leaf_node(0, depth=1)
        right = create_leaf_node(1, depth=1)
        root = create_internal_node(0, 0.5, left, right, depth=0)
        return root, left, right

    def test_routes_to_each_child(self):
        root, left, right = self._tree()
        X = np.array([[0.1], [0.4], [0.9], [0.6]])
        assert sorted(samples_reaching(root, left, X).tolist()) == [0, 1]
        assert sorted(samples_reaching(root, right, X).tolist()) == [2, 3]

    def test_root_receives_every_sample(self):
        root, _, _ = self._tree()
        X = np.array([[0.1], [0.9]])
        assert sorted(samples_reaching(root, root, X).tolist()) == [0, 1]

    def test_missing_node_returns_none(self):
        root, _, _ = self._tree()
        orphan = create_leaf_node(0, depth=1)
        assert samples_reaching(root, orphan, np.array([[0.1]])) is None

    def test_matches_by_identity_not_node_id(self):
        # Crossover grafts subtrees without renumbering, so two distinct nodes in
        # one tree can share a node_id. Matching on the id would route samples to
        # whichever happened to be visited first.
        root, left, right = self._tree()
        left.node_id = right.node_id = 7
        X = np.array([[0.1], [0.9]])
        assert samples_reaching(root, right, X).tolist() == [1]

    def test_unreachable_branch_yields_empty_index_set(self):
        root, left, right = self._tree()
        X = np.array([[0.1], [0.2]])  # nothing goes right
        assert samples_reaching(root, right, X).size == 0


class TestStepThreshold:
    def test_result_is_always_a_candidate(self):
        random.seed(0)
        candidates = np.array([0.1, 0.4, 0.9, 1.6])
        for _ in range(50):
            assert step_threshold(0.4, candidates) in candidates

    def test_step_is_never_a_no_op(self):
        random.seed(0)
        candidates = np.array([0.1, 0.4, 0.9, 1.6])
        for _ in range(50):
            assert step_threshold(0.4, candidates) != 0.4

    def test_single_candidate_is_returned_unchanged(self):
        assert step_threshold(0.4, np.array([0.7])) == 0.7


class TestInitializerUsesObservedSplits:
    def test_every_threshold_lies_between_two_observed_values(self):
        X, y = load_iris(return_X_y=True)
        random.seed(1)
        np.random.seed(1)
        initializer = TreeInitializer(
            n_features=X.shape[1],
            n_classes=3,
            max_depth=5,
            min_samples_split=5,
            min_samples_leaf=2,
            growth_stop_prob=0.05,
        )
        checked = 0
        for _ in range(30):
            tree = initializer.create_random_tree(X, y)
            for node in tree.get_internal_nodes():
                column = X[:, node.feature_idx]
                assert column.min() <= node.threshold <= column.max()
                # A threshold that separates nothing is a wasted split.
                assert 0 < int(np.sum(column <= node.threshold)) < len(column)
                checked += 1
        assert checked > 0

    def test_uniform_strategy_still_available_for_the_ablation(self):
        X, y = load_iris(return_X_y=True)
        random.seed(1)
        initializer = TreeInitializer(
            n_features=X.shape[1],
            n_classes=3,
            max_depth=4,
            min_samples_split=5,
            min_samples_leaf=2,
            split_strategy=UNIFORM_STRATEGY,
        )
        tree = initializer.create_random_tree(X, y)
        assert tree.validate()[0]

    def test_unknown_strategy_rejected_at_construction(self):
        with pytest.raises(ValueError, match="split_strategy"):
            TreeInitializer(4, 2, 3, 5, 2, split_strategy="nonsense")


class TestMutationUsesObservedSplits:
    def _setup(self, X, split_strategy=MIDPOINT_STRATEGY):
        ranges = {j: (float(X[:, j].min()), float(X[:, j].max())) for j in range(X.shape[1])}
        return Mutation(
            n_features=X.shape[1],
            feature_ranges=ranges,
            X=X,
            min_samples_leaf=2,
            split_strategy=split_strategy,
        )

    def _tree(self, X, y):
        initializer = TreeInitializer(
            n_features=X.shape[1],
            n_classes=len(np.unique(y)),
            max_depth=4,
            min_samples_split=5,
            min_samples_leaf=2,
            growth_stop_prob=0.05,
        )
        return initializer.create_random_tree(X, y)

    def test_feature_replacement_redraws_the_threshold_from_the_data(self):
        X, y = load_iris(return_X_y=True)
        random.seed(3)
        np.random.seed(3)
        mutation = self._setup(X)
        for _ in range(30):
            tree = self._tree(X, y)
            if not tree.get_internal_nodes():
                continue
            mutated = mutation.feature_replacement(tree.copy())
            for node in mutated.get_internal_nodes():
                column = X[:, node.feature_idx]
                assert column.min() <= node.threshold <= column.max()

    def test_expand_leaf_threshold_separates_the_samples_at_that_leaf(self):
        X, y = load_iris(return_X_y=True)
        random.seed(4)
        np.random.seed(4)
        mutation = self._setup(X)
        expanded = 0
        for _ in range(40):
            tree = self._tree(X, y)
            before = tree.get_num_leaves()
            mutated = mutation.expand_leaf(tree.copy())
            if mutated.get_num_leaves() == before:
                continue
            expanded += 1
            for node in mutated.get_internal_nodes():
                column = X[:, node.feature_idx]
                assert column.min() <= node.threshold <= column.max()
        assert expanded > 0, "expand_leaf never fired; test proves nothing"

    def test_threshold_perturbation_snaps_onto_an_observed_split(self):
        X, y = load_iris(return_X_y=True)
        random.seed(5)
        np.random.seed(5)
        mutation = self._setup(X)
        moved = 0
        for _ in range(40):
            tree = self._tree(X, y)
            internal = tree.get_internal_nodes()
            if not internal:
                continue
            mutated = mutation.threshold_perturbation(tree.copy())
            for original, new in zip(internal, mutated.get_internal_nodes()):
                if new.threshold != original.threshold:
                    moved += 1
                    column = X[:, new.feature_idx]
                    assert column.min() <= new.threshold <= column.max()
        assert moved > 0, "no perturbation changed a threshold; test proves nothing"

    def test_falls_back_to_feature_ranges_without_training_data(self):
        # Library users constructing Mutation the old way must keep working.
        ranges = {0: (0.0, 1.0), 1: (0.0, 1.0)}
        mutation = Mutation(n_features=2, feature_ranges=ranges)
        random.seed(6)
        tree = self._tree(*load_iris(return_X_y=True))
        mutated = mutation.feature_replacement(tree.copy())
        for node in mutated.get_internal_nodes():
            assert node.threshold is not None
