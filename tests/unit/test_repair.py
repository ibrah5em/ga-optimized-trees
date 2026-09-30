"""Constraint repair (Phase 2 item 4)."""

from pathlib import Path

import numpy as np
import pytest
import yaml

from ga_trees.ga.engine import GAConfig, GAEngine, Mutation, TreeInitializer
from ga_trees.ga.repair import count_violations, repair_constraints, repair_from_config
from ga_trees.genotype.tree_genotype import TreeGenotype, create_internal_node, create_leaf_node


def _tree(root, leaf=3, split=8):
    return TreeGenotype(
        root=root,
        n_features=1,
        n_classes=2,
        max_depth=4,
        min_samples_split=split,
        min_samples_leaf=leaf,
    )


@pytest.fixture
def data():
    X = np.arange(20, dtype=float).reshape(-1, 1)
    y = (X[:, 0] >= 10).astype(int)
    return X, y


def test_valid_split_is_left_alone(data):
    X, y = data
    tree = _tree(create_internal_node(0, 9.5, create_leaf_node(0, 1), create_leaf_node(1, 1)))
    assert count_violations(tree, X) == 0
    repair_constraints(tree, X, y)
    assert tree.get_num_nodes() == 3


def test_split_leaving_too_few_on_one_side_collapses(data):
    X, y = data
    # Two samples go right; min_samples_leaf is 3.
    tree = _tree(create_internal_node(0, 17.5, create_leaf_node(0, 1), create_leaf_node(1, 1)))
    assert count_violations(tree, X) == 1
    repair_constraints(tree, X, y)
    assert tree.root.is_leaf()
    assert tree.root.prediction in (0, 1)
    assert count_violations(tree, X) == 0


def test_unreachable_subtree_collapses_to_majority_of_its_parent(data):
    X, y = data
    # The right child tests x <= 2 but only x >= 10 reaches it: nothing goes left.
    dead = create_internal_node(0, 2.0, create_leaf_node(0, 2), create_leaf_node(0, 2))
    tree = _tree(create_internal_node(0, 9.5, create_leaf_node(0, 1), dead))
    assert count_violations(tree, X) == 1
    repair_constraints(tree, X, y)
    assert tree.get_num_nodes() == 3
    assert tree.root.right_child.is_leaf()
    assert tree.root.right_child.prediction == 1  # everything reaching it is class 1


def test_node_with_fewer_than_min_samples_split_collapses(data):
    X, y = data
    # Five samples reach the inner node, fewer than min_samples_split = 8.
    inner = create_internal_node(0, 16.5, create_leaf_node(1, 2), create_leaf_node(1, 2))
    tree = _tree(create_internal_node(0, 14.5, create_leaf_node(0, 1), inner))
    assert count_violations(tree, X) == 1
    repair_constraints(tree, X, y)
    assert tree.get_num_nodes() == 3


def test_repair_never_grows_a_tree():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(120, 4))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    init = TreeInitializer(4, 2, max_depth=5, min_samples_split=2, min_samples_leaf=1)
    for _ in range(30):
        tree = init.create_random_tree(X, y)
        before = tree.get_num_nodes()
        repair_constraints(tree, X, y, min_samples_split=20, min_samples_leaf=10)
        assert tree.get_num_nodes() <= before
        assert count_violations(tree, X, 20, 10) == 0


def test_config_gate_is_off_by_default(data):
    X, y = data
    assert repair_from_config({"min_samples_split": 8, "min_samples_leaf": 3}, X, y) is None
    fn = repair_from_config(
        {"min_samples_split": 8, "min_samples_leaf": 3, "repair_constraints": True}, X, y
    )
    assert callable(fn)


def test_pre_registered_config_leaves_repair_off():
    # The committed frontier run must stay reproducible from configs/paper.yaml.
    config = yaml.safe_load(open(Path(__file__).resolve().parents[2] / "configs" / "paper.yaml"))
    assert not config["tree"].get("repair_constraints", False)


def test_engine_offspring_satisfy_constraints_when_repair_is_on():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(150, 3))
    y = (X[:, 0] > 0).astype(int)
    tree_config = {"min_samples_split": 8, "min_samples_leaf": 3, "repair_constraints": True}
    init = TreeInitializer(3, 2, max_depth=5, min_samples_split=8, min_samples_leaf=3)
    ranges = {j: (float(X[:, j].min()), float(X[:, j].max())) for j in range(3)}
    engine = GAEngine(
        GAConfig(population_size=20, n_generations=5, mutation_prob=0.9, random_state=3),
        init,
        lambda tree, X_, y_: float(np.random.random()),
        Mutation(3, ranges, X=X, min_samples_leaf=3),
        repair=repair_from_config(tree_config, X, y),
    )
    engine.evolve(X, y, verbose=False)
    assert all(count_violations(tree, X, 8, 3) == 0 for tree in engine.population)
