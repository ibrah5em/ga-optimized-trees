import numpy as np

from ga_trees.genotype.tree_genotype import (
    create_bayesian_internal_node,
    create_bayesian_leaf_node,
    TreeGenotype,
    create_leaf_node,
)
from ga_trees.configs.bayesian_config import BayesianConfig


def test_tree_validate_and_copy():
    # Build a tiny Bayesian tree
    leaf_l = create_bayesian_leaf_node([1.0, 1.0], leaf_samples_count=10)
    leaf_r = create_bayesian_leaf_node([2.0, 1.0], leaf_samples_count=5)
    root = create_bayesian_internal_node(
        feature_idx=0,
        threshold_mean=0.5,
        threshold_std=0.2,
        threshold_dist_type="normal",
        left_child=leaf_l,
        right_child=leaf_r,
    )

    cfg = {"n_samples": 100}
    tree = TreeGenotype(root=root, n_features=1, n_classes=2, mode="bayesian", bayesian_config=cfg)
    ok, errs = tree.validate()
    assert ok, errs

    copy_tree = tree.copy()
    assert (
        isinstance(copy_tree.bayesian_config, (dict, BayesianConfig))
        or copy_tree.bayesian_config is not None
    )
