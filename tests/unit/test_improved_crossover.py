"""
Comprehensive Test for Improved Crossover
Tests:
1. Parent Map building
2. Subtree swapping
3. Structure validity after Crossover
"""

import numpy as np

from ga_trees.ga.engine import TreeInitializer
from ga_trees.ga.improved_crossover import (
    _count_subtree_nodes,
    _depth_aware_swap,
    _size_fair_swap,
    _uniform_crossover,
    build_parent_map,
    fix_depths,
    prune_to_depth,
    safe_subtree_crossover,
    validate_tree_structure,
)
from ga_trees.genotype.tree_genotype import TreeGenotype, create_internal_node, create_leaf_node


def test_parent_map():
    """Test Parent Map building."""
    print("\n" + "=" * 70)
    print("Test 1: Parent Map Building")
    print("=" * 70)

    # Create small test tree
    np.random.seed(42)
    X = np.random.rand(100, 5)
    y = np.random.randint(0, 2, 100)

    initializer = TreeInitializer(
        n_features=5, n_classes=2, max_depth=3, min_samples_split=10, min_samples_leaf=5
    )

    tree = initializer.create_random_tree(X, y)

    # Build Parent Map
    parent_map = build_parent_map(tree.root)

    print(f"✓ Tree created with {len(parent_map)} nodes")
    print(f"✓ Root parent: {parent_map[tree.root.node_id]}")

    # Verify Root has no parent
    assert parent_map[tree.root.node_id] is None, "Root should have no parent"

    # Verify every node has parent (except Root)
    for node in tree.get_all_nodes():
        if node.node_id == tree.root.node_id:
            continue
        assert node.node_id in parent_map, f"Node {node.node_id} not in parent_map"
        assert parent_map[node.node_id] is not None, f"Node {node.node_id} has no parent"

    print("✓ All nodes have correct parent references")


def test_crossover_basic():
    """Test basic Crossover."""
    print("\n" + "=" * 70)
    print("Test 2: Basic Crossover")
    print("=" * 70)

    np.random.seed(42)
    X = np.random.rand(100, 5)
    y = np.random.randint(0, 2, 100)

    initializer = TreeInitializer(
        n_features=5, n_classes=2, max_depth=4, min_samples_split=10, min_samples_leaf=5
    )

    # Create parent trees
    parent1 = initializer.create_random_tree(X, y)
    parent2 = initializer.create_random_tree(X, y)

    print(f"Parent 1: depth={parent1.get_depth()}, nodes={parent1.get_num_nodes()}")
    print(f"Parent 2: depth={parent2.get_depth()}, nodes={parent2.get_num_nodes()}")

    # Apply Crossover
    child1, child2 = safe_subtree_crossover(parent1, parent2)

    print(f"\nChild 1: depth={child1.get_depth()}, nodes={child1.get_num_nodes()}")
    print(f"Child 2: depth={child2.get_depth()}, nodes={child2.get_num_nodes()}")

    # Validation
    is_valid1, errors1 = validate_tree_structure(child1)
    is_valid2, errors2 = validate_tree_structure(child2)

    assert is_valid1, f"Child 1 has errors: {errors1}"
    assert is_valid2, f"Child 2 has errors: {errors2}"


def test_depth_fixing():
    """Test depth fixing."""
    print("\n" + "=" * 70)
    print("Test 3: Depth Fixing")
    print("=" * 70)

    np.random.seed(42)
    X = np.random.rand(100, 5)
    y = np.random.randint(0, 2, 100)

    initializer = TreeInitializer(
        n_features=5, n_classes=2, max_depth=5, min_samples_split=10, min_samples_leaf=5
    )

    tree = initializer.create_random_tree(X, y)

    # Intentionally break depths
    for node in tree.get_all_nodes():
        node.depth = 999

    print("Before fix: all depths set to 999")

    # Fix depths
    fix_depths(tree.root, 0)

    print(f"After fix: root depth = {tree.root.depth}")

    # Verification
    def check_depths(node, expected):
        if node is None:
            return True
        if node.depth != expected:
            return False
        left_ok = check_depths(node.left_child, expected + 1)
        right_ok = check_depths(node.right_child, expected + 1)
        return left_ok and right_ok

    assert check_depths(tree.root, 0), "Depth fixing failed"
    print("✓ All depths fixed correctly")


def test_pruning():
    """Test tree pruning."""
    print("\n" + "=" * 70)
    print("Test 4: Tree Pruning")
    print("=" * 70)

    np.random.seed(42)
    X = np.random.rand(100, 5)
    y = np.random.randint(0, 2, 100)

    initializer = TreeInitializer(
        n_features=5, n_classes=2, max_depth=6, min_samples_split=10, min_samples_leaf=5
    )

    tree = initializer.create_random_tree(X, y)
    original_depth = tree.get_depth()

    print(f"Original depth: {original_depth}")

    # Prune to depth=3
    tree = prune_to_depth(tree, 3)
    new_depth = tree.get_depth()

    print(f"After pruning to depth=3: {new_depth}")

    assert new_depth <= 3, f"Pruning failed: depth is {new_depth}, expected <= 3"
    print("✓ Pruning successful")


def test_multiple_crossovers():
    """Test multiple Crossovers (stress test)."""
    print("\n" + "=" * 70)
    print("Test 5: Multiple Crossovers (Stress Test)")
    print("=" * 70)

    np.random.seed(42)
    X = np.random.rand(100, 5)
    y = np.random.randint(0, 2, 100)

    initializer = TreeInitializer(
        n_features=5, n_classes=2, max_depth=4, min_samples_split=10, min_samples_leaf=5
    )

    # Create population
    population = [initializer.create_random_tree(X, y) for _ in range(10)]

    print(f"Created population of {len(population)} trees")

    # Apply crossover 20 times
    failures = 0
    for i in range(20):
        parent1 = np.random.choice(population)
        parent2 = np.random.choice(population)

        child1, child2 = safe_subtree_crossover(parent1, parent2)

        # Validation
        is_valid1, _ = validate_tree_structure(child1)
        is_valid2, _ = validate_tree_structure(child2)

        if not (is_valid1 and is_valid2):
            failures += 1

    print("Performed 20 crossovers")
    print(f"Failures: {failures}/20")
    assert failures == 0, f"{failures} crossovers produced invalid trees"
    print("✓ All crossovers successful")


def run_all_tests():
    """Run all tests."""
    print("\n" + "=" * 70)
    print("IMPROVED CROSSOVER - COMPREHENSIVE TEST SUITE")
    print("=" * 70)

    tests = [
        ("Parent Map Building", test_parent_map),
        ("Basic Crossover", test_crossover_basic),
        ("Depth Fixing", test_depth_fixing),
        ("Tree Pruning", test_pruning),
        ("Multiple Crossovers", test_multiple_crossovers),
    ]

    results = []
    for name, test_func in tests:
        try:
            success = test_func()
            results.append((name, success))
        except Exception as e:
            print(f"✗ Test '{name}' crashed: {e}")
            results.append((name, False))

    # Results summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)

    passed = sum(1 for _, success in results if success)
    total = len(results)

    for name, success in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{status}: {name}")

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\nAll tests passed! The improved crossover is working correctly.")
    else:
        print(f"\n{total - passed} test(s) failed. Check the output above.")

    return passed == total


if __name__ == "__main__":
    run_all_tests()


# ---------------------------------------------------------------------------
# §2.1 + §2.2: New crossover operator tests
# ---------------------------------------------------------------------------


def _make_depth2_tree(n_features=4):
    """Helper: create a depth-2 tree with unique feature indices."""
    ll = create_leaf_node(0, depth=2)
    lr = create_leaf_node(1, depth=2)
    rl = create_leaf_node(0, depth=2)
    rr = create_leaf_node(1, depth=2)
    left_int = create_internal_node(1, 0.3, ll, lr, depth=1)
    right_int = create_internal_node(2, 0.7, rl, rr, depth=1)
    root = create_internal_node(0, 0.5, left_int, right_int, depth=0)
    t = TreeGenotype(root=root, n_features=n_features, n_classes=2, max_depth=5)
    t._assign_node_ids(t.root, 0)
    return t


class TestDepthAwareCrossover:
    """§2.1: depth-aware subtree crossover."""

    def test_depth_aware_swap_produces_valid_result(self):
        t1 = _make_depth2_tree()
        t2 = _make_depth2_tree()
        pm1 = build_parent_map(t1.root)
        pm2 = build_parent_map(t2.root)
        nodes1 = [n for n in t1.get_all_nodes() if n.node_id != t1.root.node_id]
        nodes2 = [n for n in t2.get_all_nodes() if n.node_id != t2.root.node_id]
        result = _depth_aware_swap(nodes1, nodes2, pm1, pm2, t1.root, t2.root)
        # Should succeed (True) or gracefully fail (False) — must not raise
        assert isinstance(result, bool)

    def test_depth_aware_swap_with_only_one_node_each_side(self):
        """When all non-root nodes are at varying depths, fallback should work."""
        left = create_leaf_node(0, 1)
        right = create_leaf_node(1, 1)
        root = create_internal_node(0, 0.5, left, right, 0)
        t1 = TreeGenotype(root=root, n_features=4, n_classes=2, max_depth=4)
        t1._assign_node_ids(t1.root, 0)
        t2 = _make_depth2_tree()

        pm1 = build_parent_map(t1.root)
        pm2 = build_parent_map(t2.root)
        nodes1 = [n for n in t1.get_all_nodes() if n.node_id != t1.root.node_id]
        nodes2 = [n for n in t2.get_all_nodes() if n.node_id != t2.root.node_id]
        result = _depth_aware_swap(nodes1, nodes2, pm1, pm2, t1.root, t2.root)
        assert isinstance(result, bool)


class TestSizeFairCrossover:
    """§2.2: size-fair subtree crossover."""

    def test_count_subtree_nodes_leaf(self):
        leaf = create_leaf_node(0, 0)
        assert _count_subtree_nodes(leaf) == 1

    def test_count_subtree_nodes_depth2(self):
        t = _make_depth2_tree()
        assert _count_subtree_nodes(t.root) == 7

    def test_count_subtree_nodes_none(self):
        assert _count_subtree_nodes(None) == 0

    def test_size_fair_swap_produces_result(self):
        t1 = _make_depth2_tree()
        t2 = _make_depth2_tree()
        pm1 = build_parent_map(t1.root)
        pm2 = build_parent_map(t2.root)
        nodes1 = [n for n in t1.get_all_nodes() if n.node_id != t1.root.node_id]
        nodes2 = [n for n in t2.get_all_nodes() if n.node_id != t2.root.node_id]
        result = _size_fair_swap(nodes1, nodes2, pm1, pm2, t1.root, t2.root)
        assert isinstance(result, bool)

    def test_size_fair_falls_back_when_no_candidates(self):
        """When no node in t2 is within 2× size of selected node1, fall back to random."""
        # t1 has a large subtree, t2 has only tiny subtrees
        t1 = _make_depth2_tree()
        left = create_leaf_node(0, 1)
        right = create_leaf_node(1, 1)
        root2 = create_internal_node(0, 0.5, left, right, 0)
        t2 = TreeGenotype(root=root2, n_features=4, n_classes=2, max_depth=5)
        t2._assign_node_ids(t2.root, 0)

        pm1 = build_parent_map(t1.root)
        pm2 = build_parent_map(t2.root)
        nodes1 = [n for n in t1.get_all_nodes() if n.node_id != t1.root.node_id]
        nodes2 = [n for n in t2.get_all_nodes() if n.node_id != t2.root.node_id]
        result = _size_fair_swap(nodes1, nodes2, pm1, pm2, t1.root, t2.root)
        assert isinstance(result, bool)


class TestUniformCrossover:
    """§2.2: uniform crossover."""

    def test_uniform_crossover_preserves_structure(self):
        """After uniform crossover, both trees should still be structurally valid."""
        np.random.seed(0)
        X = np.random.rand(60, 5)
        y = np.random.randint(0, 2, 60)
        initializer = TreeInitializer(
            n_features=5, n_classes=2, max_depth=3, min_samples_split=5, min_samples_leaf=2
        )
        t1 = initializer.create_random_tree(X, y)
        t2 = initializer.create_random_tree(X, y)
        _uniform_crossover(t1, t2)
        valid1, errs1 = t1.validate()
        valid2, errs2 = t2.validate()
        assert valid1, f"t1 invalid after uniform crossover: {errs1}"
        assert valid2, f"t2 invalid after uniform crossover: {errs2}"

    def test_uniform_crossover_on_leaf_only_trees(self):
        """Uniform crossover on leaf-only trees should not crash."""
        t1 = TreeGenotype(root=create_leaf_node(0, 0), n_features=4, n_classes=2, max_depth=4)
        t2 = TreeGenotype(root=create_leaf_node(1, 0), n_features=4, n_classes=2, max_depth=4)
        _uniform_crossover(t1, t2)  # Should not raise

    def test_uniform_crossover_mismatched_structures(self):
        """Uniform crossover with trees of different depths should not crash."""
        left = create_leaf_node(0, 1)
        right = create_leaf_node(1, 1)
        root1 = create_internal_node(0, 0.5, left, right, 0)
        t1 = TreeGenotype(root=root1, n_features=4, n_classes=2, max_depth=4)
        t2 = TreeGenotype(root=create_leaf_node(0, 0), n_features=4, n_classes=2, max_depth=4)
        _uniform_crossover(t1, t2)  # Should not raise


class TestSafeSubtreeCrossoverStrategies:
    """Integration tests for safe_subtree_crossover strategy dispatch."""

    def _make_tree(self):
        np.random.seed(42)
        X = np.random.rand(80, 4)
        y = np.random.randint(0, 2, 80)
        init = TreeInitializer(
            n_features=4, n_classes=2, max_depth=4, min_samples_split=5, min_samples_leaf=2
        )
        return init.create_random_tree(X, y)

    def test_crossover_children_respect_max_depth(self):
        t1 = self._make_tree()
        t2 = self._make_tree()
        for _ in range(20):
            c1, c2 = safe_subtree_crossover(t1, t2)
            assert c1.get_depth() <= c1.max_depth
            assert c2.get_depth() <= c2.max_depth

    def test_crossover_children_are_copies(self):
        t1 = self._make_tree()
        t2 = self._make_tree()
        c1, c2 = safe_subtree_crossover(t1, t2)
        assert c1 is not t1
        assert c2 is not t2
