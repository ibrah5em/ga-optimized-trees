"""Tests for tree serialization and deserialization."""
import tempfile
from pathlib import Path

import numpy as np

from ga_trees.genotype.tree_genotype import TreeGenotype, create_internal_node, create_leaf_node


def _make_sample_tree():
    left = create_leaf_node(prediction=0, depth=1)
    right = create_leaf_node(prediction=1, depth=1)
    root = create_internal_node(
        feature_idx=2, threshold=1.5, left_child=left, right_child=right, depth=0
    )
    return TreeGenotype(root=root, n_features=4, n_classes=2, max_depth=5)


def test_to_dict_from_dict_roundtrip():
    original = _make_sample_tree()
    original.fitness_ = 0.85
    original.accuracy_ = 0.90
    data = original.to_dict()
    restored = TreeGenotype.from_dict(data)

    assert restored.n_features == original.n_features
    assert restored.n_classes == original.n_classes
    assert restored.get_depth() == original.get_depth()
    assert restored.get_num_nodes() == original.get_num_nodes()
    assert restored.fitness_ == original.fitness_


def test_to_json_from_json_roundtrip():
    original = _make_sample_tree()
    json_str = original.to_json()
    restored = TreeGenotype.from_json(json_str)

    assert restored.get_num_nodes() == original.get_num_nodes()
    assert restored.root.feature_idx == original.root.feature_idx
    assert restored.root.threshold == original.root.threshold


def test_to_json_file_roundtrip():
    original = _make_sample_tree()
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
        path = f.name
    original.to_json(filepath=path)
    restored = TreeGenotype.from_json(path)
    assert restored.get_num_nodes() == original.get_num_nodes()
    Path(path).unlink()


def test_from_dict_preserves_predictions():
    left = create_leaf_node(prediction=np.array([0.8, 0.2]), depth=1)
    right = create_leaf_node(prediction=np.array([0.1, 0.9]), depth=1)
    root = create_internal_node(0, 0.5, left, right, depth=0)
    original = TreeGenotype(root=root, n_features=4, n_classes=2)

    data = original.to_dict()
    restored = TreeGenotype.from_dict(data)

    assert isinstance(restored.root.left_child.prediction, np.ndarray)
    np.testing.assert_array_almost_equal(restored.root.left_child.prediction, [0.8, 0.2])
