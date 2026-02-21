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
    build_parent_map,
    fix_depths,
    prune_to_depth,
    safe_subtree_crossover,
    validate_tree_structure,
)
from ga_trees.genotype.tree_genotype import TreeGenotype


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