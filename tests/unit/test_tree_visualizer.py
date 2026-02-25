"""Unit tests for evaluation/tree_visualizer.py.

Covers:
- TreeVisualizer.tree_to_graphviz (with and without feature_names/class_names)
- TreeVisualizer.visualize_tree (mocked render)
- ImportError path when GRAPHVIZ_AVAILABLE = False
"""

from unittest.mock import MagicMock, patch

import pytest

import ga_trees.evaluation.tree_visualizer as viz_module
from ga_trees.evaluation.tree_visualizer import TreeVisualizer
from ga_trees.genotype.tree_genotype import TreeGenotype, create_internal_node, create_leaf_node

# ---------------------------------------------------------------------------
# Tree factories
# ---------------------------------------------------------------------------


def _simple_tree():
    """Depth-1 tree: root splits on feature 0."""
    left = create_leaf_node(prediction=0, depth=1)
    right = create_leaf_node(prediction=1, depth=1)
    root = create_internal_node(
        feature_idx=0, threshold=0.5, left_child=left, right_child=right, depth=0
    )
    return TreeGenotype(root=root, n_features=4, n_classes=2)


def _deep_tree():
    """Depth-2 tree for more thorough node traversal."""
    ll = create_leaf_node(0, depth=2)
    lr = create_leaf_node(1, depth=2)
    rl = create_leaf_node(2, depth=2)
    rr = create_leaf_node(0, depth=2)
    left_int = create_internal_node(1, 0.3, ll, lr, depth=1)
    right_int = create_internal_node(2, 0.7, rl, rr, depth=1)
    root = create_internal_node(0, 0.5, left_int, right_int, depth=0)
    return TreeGenotype(root=root, n_features=4, n_classes=3)


# ---------------------------------------------------------------------------
# Mock graphviz Digraph
# ---------------------------------------------------------------------------


def _make_mock_graphviz():
    """Return a mock graphviz module with a Digraph that records calls."""
    mock_gv = MagicMock()
    mock_digraph = MagicMock()
    mock_gv.Digraph.return_value = mock_digraph
    return mock_gv, mock_digraph


# ---------------------------------------------------------------------------
# tree_to_graphviz
# ---------------------------------------------------------------------------


class TestTreeToGraphviz:
    """Tests for TreeVisualizer.tree_to_graphviz."""

    def test_raises_when_graphviz_unavailable(self):
        """When GRAPHVIZ_AVAILABLE is False, should raise ImportError."""
        tree = _simple_tree()
        with patch.object(viz_module, "GRAPHVIZ_AVAILABLE", False):
            with pytest.raises(ImportError, match="graphviz"):
                TreeVisualizer.tree_to_graphviz(tree)

    def test_returns_digraph_when_available(self):
        """When graphviz is mocked as available, should return the Digraph."""
        tree = _simple_tree()
        mock_gv, mock_digraph = _make_mock_graphviz()
        with patch.object(viz_module, "GRAPHVIZ_AVAILABLE", True), patch.object(
            viz_module, "graphviz", mock_gv, create=True
        ):
            result = TreeVisualizer.tree_to_graphviz(tree)
        assert result is mock_digraph

    def test_adds_nodes_for_simple_tree(self):
        """For a depth-1 tree: 1 internal + 2 leaf nodes → 3 node() calls."""
        tree = _simple_tree()
        mock_gv, mock_digraph = _make_mock_graphviz()
        with patch.object(viz_module, "GRAPHVIZ_AVAILABLE", True), patch.object(
            viz_module, "graphviz", mock_gv, create=True
        ):
            TreeVisualizer.tree_to_graphviz(tree)
        assert mock_digraph.node.call_count == 3

    def test_adds_edges_for_simple_tree(self):
        """Depth-1 tree has 2 edges (root→left, root→right)."""
        tree = _simple_tree()
        mock_gv, mock_digraph = _make_mock_graphviz()
        with patch.object(viz_module, "GRAPHVIZ_AVAILABLE", True), patch.object(
            viz_module, "graphviz", mock_gv, create=True
        ):
            TreeVisualizer.tree_to_graphviz(tree)
        assert mock_digraph.edge.call_count == 2

    def test_uses_feature_names_when_provided(self):
        """Internal node label should contain the feature name from the list."""
        tree = _simple_tree()
        feature_names = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
        mock_gv, mock_digraph = _make_mock_graphviz()
        with patch.object(viz_module, "GRAPHVIZ_AVAILABLE", True), patch.object(
            viz_module, "graphviz", mock_gv, create=True
        ):
            TreeVisualizer.tree_to_graphviz(tree, feature_names=feature_names)

        # Check that one of the node calls uses "sepal_length"
        all_labels = [str(c) for c in mock_digraph.node.call_args_list]
        assert any("sepal_length" in label for label in all_labels)

    def test_uses_default_feature_label_without_names(self):
        """Without feature_names, internal node label should contain 'X['."""
        tree = _simple_tree()
        mock_gv, mock_digraph = _make_mock_graphviz()
        with patch.object(viz_module, "GRAPHVIZ_AVAILABLE", True), patch.object(
            viz_module, "graphviz", mock_gv, create=True
        ):
            TreeVisualizer.tree_to_graphviz(tree)

        all_labels = [str(c) for c in mock_digraph.node.call_args_list]
        assert any("X[" in label for label in all_labels)

    def test_deep_tree_traversal(self):
        """Depth-2 tree: 3 internal + 4 leaf nodes → 7 node calls, 6 edge calls."""
        tree = _deep_tree()
        mock_gv, mock_digraph = _make_mock_graphviz()
        with patch.object(viz_module, "GRAPHVIZ_AVAILABLE", True), patch.object(
            viz_module, "graphviz", mock_gv, create=True
        ):
            TreeVisualizer.tree_to_graphviz(tree)
        assert mock_digraph.node.call_count == 7
        assert mock_digraph.edge.call_count == 6

    def test_leaf_node_uses_green_fill(self):
        """Leaf nodes should be filled with '#90EE90'."""
        tree = _simple_tree()
        mock_gv, mock_digraph = _make_mock_graphviz()
        with patch.object(viz_module, "GRAPHVIZ_AVAILABLE", True), patch.object(
            viz_module, "graphviz", mock_gv, create=True
        ):
            TreeVisualizer.tree_to_graphviz(tree)

        all_args = [str(c) for c in mock_digraph.node.call_args_list]
        assert any("#90EE90" in a for a in all_args)

    def test_internal_node_uses_blue_fill(self):
        """Internal nodes should be filled with '#87CEEB'."""
        tree = _simple_tree()
        mock_gv, mock_digraph = _make_mock_graphviz()
        with patch.object(viz_module, "GRAPHVIZ_AVAILABLE", True), patch.object(
            viz_module, "graphviz", mock_gv, create=True
        ):
            TreeVisualizer.tree_to_graphviz(tree)

        all_args = [str(c) for c in mock_digraph.node.call_args_list]
        assert any("#87CEEB" in a for a in all_args)


# ---------------------------------------------------------------------------
# visualize_tree
# ---------------------------------------------------------------------------


class TestVisualizeTree:
    """Tests for TreeVisualizer.visualize_tree."""

    def test_calls_render(self):
        """visualize_tree should call dot.render()."""
        tree = _simple_tree()
        mock_gv, mock_digraph = _make_mock_graphviz()
        with patch.object(viz_module, "GRAPHVIZ_AVAILABLE", True), patch.object(
            viz_module, "graphviz", mock_gv, create=True
        ):
            TreeVisualizer.visualize_tree(tree)
        mock_digraph.render.assert_called_once()

    def test_render_called_with_correct_format(self):
        """render should be called with format='png' and cleanup=True."""
        tree = _simple_tree()
        mock_gv, mock_digraph = _make_mock_graphviz()
        with patch.object(viz_module, "GRAPHVIZ_AVAILABLE", True), patch.object(
            viz_module, "graphviz", mock_gv, create=True
        ):
            TreeVisualizer.visualize_tree(tree, save_path="/tmp/test_tree")  # nosec B108
        mock_digraph.render.assert_called_once_with(  # nosec B108
            "/tmp/test_tree", format="png", cleanup=True
        )

    def test_raises_when_graphviz_unavailable(self):
        """If GRAPHVIZ_AVAILABLE is False, should raise ImportError."""
        tree = _simple_tree()
        with patch.object(viz_module, "GRAPHVIZ_AVAILABLE", False):
            with pytest.raises(ImportError):
                TreeVisualizer.visualize_tree(tree)
