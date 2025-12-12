"""
Improved Crossover Implementation with Parent Tracking

Complete solution for the Crossover problem with Parent Nodes tracking

Stages:
1. Build Parent Map for the tree
2. Safe Crossover with reference updates
3. Post-Crossover validation
"""

import copy
import random
from typing import Dict, Optional, Tuple

import numpy as np

from ga_trees.genotype.tree_genotype import Node, TreeGenotype, create_leaf_node

# ============================================================================
# Stage 1: Building Parent Map
# ============================================================================


def build_parent_map(root: Node) -> Dict[int, Optional[Node]]:
    """
    Build a map linking each Node to its Parent.

    Args:
        root: Root Node of the tree

    Returns:
        Dictionary: {node_id: parent_node}
        Root will have None as parent
    """
    parent_map = {root.node_id: None}  # Root has no parent

    def traverse(node: Node, parent: Optional[Node]):
        if node is None:
            return

        if parent is not None:
            parent_map[node.node_id] = parent

        if node.left_child:
            traverse(node.left_child, node)
        if node.right_child:
            traverse(node.right_child, node)

    traverse(root, None)
    return parent_map


def get_node_by_id(root: Node, node_id: int) -> Optional[Node]:
    """
    Find Node by ID.

    Args:
        root: Root Node
        node_id: Required ID

    Returns:
        Node if found, None if not found
    """
    if root is None:
        return None

    if root.node_id == node_id:
        return root

    # Search in left branch
    left_result = get_node_by_id(root.left_child, node_id)
    if left_result:
        return left_result

    # Search in right branch
    return get_node_by_id(root.right_child, node_id)


# ============================================================================
# Stage 2: Safe Crossover
# ============================================================================


def safe_subtree_crossover(
    parent1: TreeGenotype, parent2: TreeGenotype
) -> Tuple[TreeGenotype, TreeGenotype]:
    """
    Improved Subtree Crossover with Parent Tracking.

    Steps:
    1. Copy Parents
    2. Build Parent Maps
    3. Select nodes for swapping
    4. Swap subtrees safely
    5. Fix depths
    6. Validation

    Args:
        parent1: First tree
        parent2: Second tree

    Returns:
        (child1, child2): Resulting children
    """
    # 1. Deep copy trees
    child1 = parent1.copy()
    child2 = parent2.copy()

    # 2. Build Parent Maps
    parent_map1 = build_parent_map(child1.root)
    parent_map2 = build_parent_map(child2.root)

    # 3. Collect all nodes (except Root to avoid problems)
    all_nodes1 = [n for n in child1.get_all_nodes() if n.node_id != child1.root.node_id]
    all_nodes2 = [n for n in child2.get_all_nodes() if n.node_id != child2.root.node_id]

    # If not enough nodes, return copies without modification
    if len(all_nodes1) < 1 or len(all_nodes2) < 1:
        return child1, child2

    # 4. Randomly select nodes for swapping
    node1 = random.choice(all_nodes1)
    node2 = random.choice(all_nodes2)

    # 5. Swap subtrees
    swap_success = swap_subtrees(node1, node2, parent_map1, parent_map2, child1.root, child2.root)

    if not swap_success:
        # If swap failed, return original copies
        return child1, child2

    # 6. Fix depths after swapping
    fix_depths(child1.root, 0)
    fix_depths(child2.root, 0)

    # 7. Pruning if trees exceed max_depth
    if child1.get_depth() > child1.max_depth:
        child1 = prune_to_depth(child1, child1.max_depth)
    if child2.get_depth() > child2.max_depth:
        child2 = prune_to_depth(child2, child2.max_depth)

    # 8. Reset IDs
    child1._assign_node_ids(child1.root, 0)
    child2._assign_node_ids(child2.root, 0)

    # 9. Validation (optional - for development)
    # is_valid1, errors1 = child1.validate()
    # is_valid2, errors2 = child2.validate()

    return child1, child2


def swap_subtrees(
    node1: Node,
    node2: Node,
    parent_map1: Dict[int, Optional[Node]],
    parent_map2: Dict[int, Optional[Node]],
    root1: Node,
    root2: Node,
) -> bool:
    """
    Swap subtrees between node1 and node2 while updating parents.

    Args:
        node1: Node from first tree
        node2: Node from second tree
        parent_map1: Parent map for first tree
        parent_map2: Parent map for second tree
        root1: Root of first tree
        root2: Root of second tree

    Returns:
        True if swap successful, False if failed
    """
    try:
        # Get parents
        parent1 = parent_map1.get(node1.node_id)
        parent2 = parent_map2.get(node2.node_id)

        # Copy subtrees before swapping
        subtree1_copy = node1.copy()
        subtree2_copy = node2.copy()

        # Determine if node1 is left or right child of parent1
        if parent1 is not None:
            if parent1.left_child and parent1.left_child.node_id == node1.node_id:
                parent1.left_child = subtree2_copy
            elif parent1.right_child and parent1.right_child.node_id == node1.node_id:
                parent1.right_child = subtree2_copy
            else:
                return False
        else:
            # node1 is root - cannot swap
            return False

        # Determine if node2 is left or right child of parent2
        if parent2 is not None:
            if parent2.left_child and parent2.left_child.node_id == node2.node_id:
                parent2.left_child = subtree1_copy
            elif parent2.right_child and parent2.right_child.node_id == node2.node_id:
                parent2.right_child = subtree1_copy
            else:
                return False
        else:
            # node2 is root - cannot swap
            return False

        return True

    except Exception as e:
        print(f"Swap failed: {e}")
        return False


# ============================================================================
# Stage 3: Utility Functions
# ============================================================================


def fix_depths(node: Node, depth: int):
    """
    Recursively fix depth values after Crossover.

    Args:
        node: Current Node
        depth: Correct depth for the Node
    """
    if node is None:
        return

    node.depth = depth

    if node.left_child:
        fix_depths(node.left_child, depth + 1)
    if node.right_child:
        fix_depths(node.right_child, depth + 1)


def prune_to_depth(tree: TreeGenotype, max_depth: int) -> TreeGenotype:
    """
    Prune tree to specific depth.

    Args:
        tree: Tree to prune
        max_depth: Maximum allowed depth

    Returns:
        Pruned tree
    """

    def prune_node(node: Node, depth: int) -> Node:
        if node is None:
            return None

        # If we reached maximum depth, convert to leaf
        if depth >= max_depth:
            if node.is_leaf():
                return node
            else:
                # Convert internal node to leaf
                return create_leaf_node(0, depth)

        # If leaf, return as is
        if node.is_leaf():
            return node

        # Recursively prune children
        node.left_child = prune_node(node.left_child, depth + 1)
        node.right_child = prune_node(node.right_child, depth + 1)

        return node

    tree.root = prune_node(tree.root, 0)
    return tree


def validate_tree_structure(tree: TreeGenotype) -> Tuple[bool, list]:
    """
    Validate tree structure after Crossover.

    Args:
        tree: Tree to check

    Returns:
        (is_valid, list_of_errors)
    """
    errors = []

    # Check that every internal node has two children
    def check_structure(node: Node):
        if node is None:
            return

        if node.is_internal():
            if node.left_child is None:
                errors.append(f"Internal node {node.node_id} missing left child")
            if node.right_child is None:
                errors.append(f"Internal node {node.node_id} missing right child")
        else:
            if node.left_child is not None or node.right_child is not None:
                errors.append(f"Leaf node {node.node_id} has children")

        if node.left_child:
            check_structure(node.left_child)
        if node.right_child:
            check_structure(node.right_child)

    check_structure(tree.root)

    # Check depths
    def check_depths(node: Node, expected_depth: int):
        if node is None:
            return
        if node.depth != expected_depth:
            errors.append(
                f"Node {node.node_id} has wrong depth: {node.depth} " f"(expected {expected_depth})"
            )
        if node.left_child:
            check_depths(node.left_child, expected_depth + 1)
        if node.right_child:
            check_depths(node.right_child, expected_depth + 1)

    check_depths(tree.root, 0)

    return (len(errors) == 0, errors)


# ============================================================================
# Usage example
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Improved Crossover - Test")
    print("=" * 70)

    # This is a demonstration example - needs actual TreeGenotype to run
    print(
        """
    Usage in GAEngine:
    
    # Replace this code in engine.py:
    
    class Crossover:
        @staticmethod
        def subtree_crossover(parent1, parent2):
            return safe_subtree_crossover(parent1, parent2)
    
   """
    )
