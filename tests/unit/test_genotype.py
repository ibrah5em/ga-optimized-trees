"""Unit tests for tree genotype."""

from ga_trees.genotype.tree_genotype import TreeGenotype, create_internal_node, create_leaf_node


class TestNode:
    """Test Node class."""

    def test_create_leaf(self):
        """Test leaf node creation."""
        leaf = create_leaf_node(prediction=1, depth=0)
        assert leaf.is_leaf()
        assert not leaf.is_internal()
        assert leaf.prediction == 1
        assert leaf.depth == 0

    def test_create_internal(self):
        """Test internal node creation."""
        left = create_leaf_node(0, depth=1)
        right = create_leaf_node(1, depth=1)
        internal = create_internal_node(
            feature_idx=0, threshold=0.5, left_child=left, right_child=right, depth=0
        )

        assert internal.is_internal()
        assert not internal.is_leaf()
        assert internal.feature_idx == 0
        assert internal.threshold == 0.5
        assert internal.left_child == left
        assert internal.right_child == right

    def test_node_height(self):
        """Test height calculation."""
        # Single leaf
        leaf = create_leaf_node(0, 0)
        assert leaf.get_height() == 0

        # Tree with depth 2
        left = create_leaf_node(0, 2)
        right = create_leaf_node(1, 2)
        internal1 = create_internal_node(1, 0.5, left, right, 1)
        leaf2 = create_leaf_node(0, 2)
        root = create_internal_node(0, 0.3, internal1, leaf2, 0)

        assert root.get_height() == 2

    def test_count_nodes(self):
        """Test node counting."""
        leaf = create_leaf_node(0, 0)
        assert leaf.count_nodes() == 1

        left = create_leaf_node(0, 1)
        right = create_leaf_node(1, 1)
        root = create_internal_node(0, 0.5, left, right, 0)
        assert root.count_nodes() == 3


class TestTreeGenotype:
    """Test TreeGenotype class."""

    def test_create_simple_tree(self):
        """Test creating a simple tree."""
        left = create_leaf_node(0, 1)
        right = create_leaf_node(1, 1)
        root = create_internal_node(0, 0.5, left, right, 0)

        tree = TreeGenotype(
            root=root, n_features=4, n_classes=2, task_type="classification", max_depth=5
        )

        assert tree.get_depth() == 1
        assert tree.get_num_nodes() == 3
        assert tree.get_num_leaves() == 2

    def test_get_all_nodes(self):
        """Test getting all nodes."""
        left = create_leaf_node(0, 1)
        right = create_leaf_node(1, 1)
        root = create_internal_node(0, 0.5, left, right, 0)

        tree = TreeGenotype(root=root, n_features=4, n_classes=2)
        nodes = tree.get_all_nodes()

        assert len(nodes) == 3
        assert root in nodes
        assert left in nodes
        assert right in nodes

    def test_features_used(self):
        """Test feature tracking."""
        left = create_leaf_node(0, 2)
        right = create_leaf_node(1, 2)
        internal = create_internal_node(1, 0.5, left, right, 1)
        leaf = create_leaf_node(0, 1)
        root = create_internal_node(0, 0.3, internal, leaf, 0)

        tree = TreeGenotype(root=root, n_features=4, n_classes=2)
        features = tree.get_features_used()

        assert features == {0, 1}
        assert tree.get_num_features_used() == 2

    def test_tree_balance(self):
        """Test balance calculation."""
        # Perfectly balanced tree
        l1 = create_leaf_node(0, 1)
        l2 = create_leaf_node(1, 1)
        root = create_internal_node(0, 0.5, l1, l2, 0)
        tree = TreeGenotype(root=root, n_features=4, n_classes=2)

        balance = tree.get_tree_balance()
        assert 0.9 <= balance <= 1.0  # Should be near perfect

    def test_validation_depth(self):
        """Test depth validation."""
        # Create tree that violates depth
        nodes = create_leaf_node(0, 0)
        for i in range(10):
            nodes = create_internal_node(0, 0.5, nodes, create_leaf_node(0, i + 1), i)

        tree = TreeGenotype(root=nodes, n_features=4, n_classes=2, max_depth=5)
        valid, errors = tree.validate()

        assert not valid
        assert any("depth" in str(e).lower() for e in errors)

    def test_to_rules(self):
        """Test rule extraction."""
        left = create_leaf_node(0, 1)
        right = create_leaf_node(1, 1)
        root = create_internal_node(0, 0.5, left, right, 0)

        tree = TreeGenotype(root=root, n_features=4, n_classes=2)
        rules = tree.to_rules()

        assert len(rules) == 2
        assert all("IF" in rule and "THEN" in rule for rule in rules)


# ---------------------------------------------------------------------------
# to_dict / _node_to_dict  (lines 279-304)
# ---------------------------------------------------------------------------


class TestToDict:
    def _depth1_tree(self):
        left = create_leaf_node(0, depth=1)
        right = create_leaf_node(1, depth=1)
        root = create_internal_node(0, 0.5, left, right, depth=0)
        return TreeGenotype(root=root, n_features=4, n_classes=2, max_depth=5)

    def test_returns_dict(self):
        tree = self._depth1_tree()
        result = tree.to_dict()
        assert isinstance(result, dict)

    def test_contains_required_top_level_keys(self):
        tree = self._depth1_tree()
        result = tree.to_dict()
        for key in ("n_features", "n_classes", "task_type", "max_depth", "root", "metadata"):
            assert key in result

    def test_metadata_keys_present(self):
        tree = self._depth1_tree()
        meta = tree.to_dict()["metadata"]
        for key in ("depth", "num_nodes", "num_leaves", "features_used", "balance"):
            assert key in meta

    def test_root_is_dict(self):
        tree = self._depth1_tree()
        root_dict = tree.to_dict()["root"]
        assert isinstance(root_dict, dict)
        assert root_dict["node_type"] == "internal"

    def test_internal_node_has_children(self):
        tree = self._depth1_tree()
        root_dict = tree.to_dict()["root"]
        assert "left_child" in root_dict
        assert "right_child" in root_dict

    def test_leaf_node_has_prediction(self):
        tree = self._depth1_tree()
        root_dict = tree.to_dict()["root"]
        left = root_dict["left_child"]
        assert left["node_type"] == "leaf"
        assert "prediction" in left

    def test_ndarray_prediction_serialized_as_list(self):
        """Leaf with ndarray prediction → to_dict should convert to list."""
        import numpy as np

        root = create_leaf_node(np.array([0.3, 0.7]), depth=0)
        tree = TreeGenotype(root=root, n_features=4, n_classes=2)
        root_dict = tree.to_dict()["root"]
        assert isinstance(root_dict["prediction"], list)

    def test_n_features_matches(self):
        tree = self._depth1_tree()
        assert tree.to_dict()["n_features"] == 4

    def test_none_node_returns_none(self):
        tree = self._depth1_tree()
        assert tree._node_to_dict(None) is None


# ---------------------------------------------------------------------------
# to_rules / _extract_rules  (lines 315, 320, 323)
# ---------------------------------------------------------------------------


class TestToRules:
    def test_leaf_only_tree_returns_one_rule_with_true(self):
        """A single-leaf tree: no conditions → 'IF True THEN class=...'"""
        root = create_leaf_node(prediction=0, depth=0)
        tree = TreeGenotype(root=root, n_features=4, n_classes=2)
        rules = tree.to_rules()
        assert len(rules) == 1
        assert "True" in rules[0]

    def test_depth1_tree_returns_two_rules(self):
        left = create_leaf_node(0, 1)
        right = create_leaf_node(1, 1)
        root = create_internal_node(0, 0.5, left, right, 0)
        tree = TreeGenotype(root=root, n_features=4, n_classes=2)
        rules = tree.to_rules()
        assert len(rules) == 2

    def test_rules_contain_if_then(self):
        left = create_leaf_node(0, 1)
        right = create_leaf_node(1, 1)
        root = create_internal_node(0, 0.5, left, right, 0)
        tree = TreeGenotype(root=root, n_features=4, n_classes=2)
        rules = tree.to_rules()
        for rule in rules:
            assert rule.startswith("IF ")
            assert "THEN class=" in rule

    def test_ndarray_prediction_uses_argmax(self):
        """When prediction is ndarray, rule should show argmax, not the array."""
        import numpy as np

        root = create_leaf_node(np.array([0.1, 0.9]), depth=0)
        tree = TreeGenotype(root=root, n_features=4, n_classes=2)
        rules = tree.to_rules()
        # argmax of [0.1, 0.9] = 1
        assert "class=1" in rules[0]

    def test_rules_conditions_mention_feature_index(self):
        """Internal node conditions should reference 'X[...]'."""
        left = create_leaf_node(0, 1)
        right = create_leaf_node(1, 1)
        root = create_internal_node(2, 0.75, left, right, 0)
        tree = TreeGenotype(root=root, n_features=4, n_classes=2)
        rules = tree.to_rules()
        all_rules = " ".join(rules)
        assert "X[2]" in all_rules


# ---------------------------------------------------------------------------
# _check_structure false paths  (lines 228, 233, 237, 242, 245, 247)
# ---------------------------------------------------------------------------


class TestCheckStructure:
    def test_leaf_with_child_is_invalid(self):
        """A leaf that has a left child fails structure check."""
        leaf = create_leaf_node(0, depth=1)
        # Manually add a child to make it invalid
        leaf.left_child = create_leaf_node(0, depth=2)
        root = create_internal_node(0, 0.5, leaf, create_leaf_node(1, 1), depth=0)
        tree = TreeGenotype(root=root, n_features=4, n_classes=2)
        valid, errors = tree.validate()
        assert not valid

    def test_internal_without_left_child_is_invalid(self):
        """An internal node missing its left child fails structure check."""
        left = create_leaf_node(0, 1)
        right = create_leaf_node(1, 1)
        root = create_internal_node(0, 0.5, left, right, depth=0)
        # Manually remove left child to create invalid structure
        root.left_child = None
        tree = TreeGenotype.__new__(TreeGenotype)
        tree.root = root
        tree.n_features = 4
        tree.n_classes = 2
        tree.task_type = "classification"
        tree.max_depth = 5
        tree.min_samples_split = 10
        tree.min_samples_leaf = 5
        tree.max_features = 4
        tree.fitness_ = None
        tree.accuracy_ = None
        tree.interpretability_ = None
        valid = tree._check_structure(root)
        assert not valid

    def test_depth_inconsistency_is_invalid(self):
        """Child depth != parent depth + 1 fails structure check."""
        left = create_leaf_node(0, depth=99)  # Wrong depth
        right = create_leaf_node(1, depth=1)
        root = create_internal_node(0, 0.5, left, right, depth=0)
        # Use _check_structure directly
        tree = TreeGenotype(root=root, n_features=4, n_classes=2)
        valid = tree._check_structure(root)
        assert not valid


# ---------------------------------------------------------------------------
# validate() error paths  (lines 208, 212, 217)
# ---------------------------------------------------------------------------


class TestValidateErrorPaths:
    def test_invalid_feature_index_detected(self):
        """Feature index out of range should produce a validation error."""
        left = create_leaf_node(0, 1)
        right = create_leaf_node(1, 1)
        # feature_idx=10 is out of range for n_features=4
        root = create_internal_node(10, 0.5, left, right, depth=0)
        tree = TreeGenotype(root=root, n_features=4, n_classes=2)
        valid, errors = tree.validate()
        assert not valid
        assert any("feature" in e.lower() for e in errors)

    def test_missing_threshold_detected(self):
        """Internal node with threshold=None should produce a validation error."""
        left = create_leaf_node(0, 1)
        right = create_leaf_node(1, 1)
        root = create_internal_node(0, 0.5, left, right, depth=0)
        root.threshold = None  # Make invalid
        tree = TreeGenotype(root=root, n_features=4, n_classes=2)
        valid, errors = tree.validate()
        assert not valid
        assert any("threshold" in e.lower() for e in errors)

    def test_leaf_with_none_prediction_detected(self):
        """Leaf node with prediction=None should produce a validation error."""
        left = create_leaf_node(None, 1)  # None prediction
        right = create_leaf_node(1, 1)
        root = create_internal_node(0, 0.5, left, right, depth=0)
        tree = TreeGenotype(root=root, n_features=4, n_classes=2)
        valid, errors = tree.validate()
        assert not valid
        assert any("prediction" in e.lower() for e in errors)
