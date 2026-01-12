import numpy as np
from numpy.random import default_rng

from ga_trees.genotype.tree_genotype import (
    create_bayesian_internal_node,
    create_bayesian_leaf_node,
    create_leaf_node,
    get_soft_decision_prob,
    sample_leaf_distribution,
    sample_threshold,
)


def test_sample_threshold_and_leaf_sampling():
    rng = default_rng(42)
    # Threshold sampling normal
    node = create_bayesian_internal_node(
        feature_idx=0, threshold_mean=0.0, threshold_std=1.0, threshold_dist_type="normal"
    )
    val = sample_threshold(node, rng=rng)
    assert isinstance(val, float)

    # Leaf Dirichlet sampling
    leaf = create_bayesian_leaf_node([2.0, 1.0], leaf_samples_count=5)
    vec = sample_leaf_distribution(leaf, n_classes=2, rng=rng)
    assert np.isclose(vec.sum(), 1.0)
    assert vec.shape == (2,)


def test_get_soft_decision_prob_normal():
    # Analytical normal test
    left = create_leaf_node(0)
    right = create_leaf_node(1)
    node = create_bayesian_internal_node(
        feature_idx=0,
        threshold_mean=0.5,
        threshold_std=0.1,
        threshold_dist_type="normal",
        left_child=left,
        right_child=right,
    )
    x_row = np.array([0.6])
    p = get_soft_decision_prob(node, x_row)
    assert 0.0 <= p <= 1.0
