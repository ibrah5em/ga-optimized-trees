"""
Decision Tree Genotype Representation

This module defines the core tree structure used by the genetic algorithm.
Trees are represented as binary decision trees with constrained structure.
"""

import copy
from dataclasses import dataclass, field
from typing import List, Literal, Optional, Tuple, Union

import numpy as np


@dataclass
class Node:
    """A node in the decision tree (internal or leaf)."""

    # Node type
    node_type: Literal["internal", "leaf"] = "leaf"

    # For internal nodes
    feature_idx: Optional[int] = None
    threshold: Optional[float] = None
    operator: Literal["<=", ">"] = "<="

    # Children (for internal nodes)
    left_child: Optional["Node"] = None
    right_child: Optional["Node"] = None

    # For leaf nodes
    prediction: Optional[Union[int, float, np.ndarray]] = None

    # Metadata
    depth: int = 0
    node_id: int = 0
    samples_count: int = 0

    def is_leaf(self) -> bool:
        """Check if this is a leaf node."""
        return self.node_type == "leaf"

    def is_internal(self) -> bool:
        """Check if this is an internal node."""
        return self.node_type == "internal"

    def get_height(self) -> int:
        """Get the height of subtree rooted at this node."""
        if self.is_leaf():
            return 0
        left_h = self.left_child.get_height() if self.left_child else 0
        right_h = self.right_child.get_height() if self.right_child else 0
        return 1 + max(left_h, right_h)

    def count_nodes(self) -> int:
        """Count total nodes in subtree."""
        if self.is_leaf():
            return 1
        left_count = self.left_child.count_nodes() if self.left_child else 0
        right_count = self.right_child.count_nodes() if self.right_child else 0
        return 1 + left_count + right_count

    def get_leaf_depths(self) -> List[int]:
        """Get depths of all leaves in subtree."""
        if self.is_leaf():
            return [self.depth]
        depths = []
        if self.left_child:
            depths.extend(self.left_child.get_leaf_depths())
        if self.right_child:
            depths.extend(self.right_child.get_leaf_depths())
        return depths

    def get_features_used(self) -> set:
        """Get set of features used in subtree."""
        if self.is_leaf():
            return set()
        features = {self.feature_idx} if self.feature_idx is not None else set()
        if self.left_child:
            features.update(self.left_child.get_features_used())
        if self.right_child:
            features.update(self.right_child.get_features_used())
        return features

    def copy(self) -> "Node":
        """Create a deep copy of this node and its subtree."""
        return copy.deepcopy(self)


@dataclass
class TreeGenotype:
    """
    Genotype representation of a decision tree.

    This class represents the internal structure of a decision tree
    that can be evolved by the genetic algorithm.
    """

    root: Node
    n_features: int
    n_classes: int
    task_type: Literal["classification", "regression"] = "classification"

    # Constraints
    max_depth: int = 5
    min_samples_split: int = 10
    min_samples_leaf: int = 5
    max_features: Optional[int] = None

    # Metadata
    fitness_: Optional[float] = None
    accuracy_: Optional[float] = None
    interpretability_: Optional[float] = None

    def __post_init__(self):
        """Initialize derived attributes."""
        if self.max_features is None:
            self.max_features = self.n_features
        self._assign_node_ids(self.root, 0)

    def _assign_node_ids(self, node: Node, next_id: int) -> int:
        """Assign unique IDs to all nodes."""
        if node is None:
            return next_id
        node.node_id = next_id
        next_id += 1
        if node.left_child:
            next_id = self._assign_node_ids(node.left_child, next_id)
        if node.right_child:
            next_id = self._assign_node_ids(node.right_child, next_id)
        return next_id

    def get_depth(self) -> int:
        """Get maximum depth of the tree."""
        return self.root.get_height()

    def get_num_nodes(self) -> int:
        """Get total number of nodes."""
        return self.root.count_nodes()

    def get_num_leaves(self) -> int:
        """Get number of leaf nodes."""
        return len(self.get_all_leaves())

    def get_all_nodes(self) -> List[Node]:
        """Get list of all nodes in tree."""
        nodes = []
        self._collect_nodes(self.root, nodes)
        return nodes

    def _collect_nodes(self, node: Node, nodes: List[Node]):
        """Helper to collect all nodes."""
        if node is None:
            return
        nodes.append(node)
        if node.left_child:
            self._collect_nodes(node.left_child, nodes)
        if node.right_child:
            self._collect_nodes(node.right_child, nodes)

    def get_all_leaves(self) -> List[Node]:
        """Get list of all leaf nodes."""
        return [n for n in self.get_all_nodes() if n.is_leaf()]

    def get_internal_nodes(self) -> List[Node]:
        """Get list of all internal nodes."""
        return [n for n in self.get_all_nodes() if n.is_internal()]

    def get_features_used(self) -> set:
        """Get set of all features used in tree."""
        return self.root.get_features_used()

    def get_num_features_used(self) -> int:
        """Get count of unique features used."""
        return len(self.get_features_used())

    def get_tree_balance(self) -> float:
        """
        Calculate tree balance metric.
        Returns value in [0, 1] where 1 is perfectly balanced.
        """
        leaf_depths = self.root.get_leaf_depths()
        if len(leaf_depths) <= 1:
            return 1.0
        depth_std = np.std(leaf_depths)
        max_depth = self.get_depth()
        if max_depth == 0:
            return 1.0
        return 1.0 - min(depth_std / max_depth, 1.0)

    def validate(self) -> Tuple[bool, List[str]]:
        """
        Validate tree structure and constraints.

        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []

        # Check depth constraint
        if self.get_depth() > self.max_depth:
            errors.append(f"Tree depth {self.get_depth()} exceeds max_depth {self.max_depth}")

        # Check feature indices
        for node in self.get_internal_nodes():
            if node.feature_idx is not None:
                if node.feature_idx < 0 or node.feature_idx >= self.n_features:
                    errors.append(
                        f"Invalid feature index {node.feature_idx} at node {node.node_id}"
                    )
            if node.threshold is None:
                errors.append(f"Internal node {node.node_id} missing threshold")

        # Check leaf nodes have predictions
        for node in self.get_all_leaves():
            if node.prediction is None:
                errors.append(f"Leaf node {node.node_id} missing prediction")

        # Check tree structure
        if not self._check_structure(self.root):
            errors.append("Tree structure is inconsistent")

        return (len(errors) == 0, errors)

    def _check_structure(self, node: Node) -> bool:
        """Check tree structure consistency."""
        if node is None:
            return True

        if node.is_leaf():
            # Leaves should not have children
            if node.left_child is not None or node.right_child is not None:
                return False
        else:
            # Internal nodes must have both children
            if node.left_child is None or node.right_child is None:
                return False
            # Check depth consistency
            if node.left_child.depth != node.depth + 1:
                return False
            if node.right_child.depth != node.depth + 1:
                return False
            # Recursively check children
            if not self._check_structure(node.left_child):
                return False
            if not self._check_structure(node.right_child):
                return False

        return True

    def copy(self) -> "TreeGenotype":
        """Create a deep copy of this tree."""
        return copy.deepcopy(self)

    def to_dict(self) -> dict:
        """Convert tree to dictionary representation."""
        return {
            "n_features": self.n_features,
            "n_classes": self.n_classes,
            "task_type": self.task_type,
            "max_depth": self.max_depth,
            "min_samples_split": self.min_samples_split,
            "min_samples_leaf": self.min_samples_leaf,
            "root": self._node_to_dict(self.root),
            "metadata": {
                "depth": self.get_depth(),
                "num_nodes": self.get_num_nodes(),
                "num_leaves": self.get_num_leaves(),
                "features_used": list(self.get_features_used()),
                "balance": self.get_tree_balance(),
                "fitness": self.fitness_,
                "accuracy": self.accuracy_,
                "interpretability": self.interpretability_,
            },
        }

    def _node_to_dict(self, node: Node) -> dict:
        """Convert node to dictionary."""
        if node is None:
            return None

        result = {
            "node_type": node.node_type,
            "node_id": node.node_id,
            "depth": node.depth,
        }

        if node.is_internal():
            result.update(
                {
                    "feature_idx": node.feature_idx,
                    "threshold": float(node.threshold) if node.threshold is not None else None,
                    "operator": node.operator,
                    "left_child": self._node_to_dict(node.left_child),
                    "right_child": self._node_to_dict(node.right_child),
                }
            )
        else:
            if isinstance(node.prediction, np.ndarray):
                result["prediction"] = node.prediction.tolist()
            else:
                result["prediction"] = node.prediction

        return result

    def to_rules(self) -> List[str]:
        """Extract human-readable rules from tree."""
        rules = []
        self._extract_rules(self.root, [], rules)
        return rules

    def _extract_rules(self, node: Node, conditions: List[str], rules: List[str]):
        """Helper to extract rules recursively."""
        if node is None:
            return

        if node.is_leaf():
            rule = " AND ".join(conditions)
            if not rule:
                rule = "True"
            pred = node.prediction
            if isinstance(pred, np.ndarray):
                pred = np.argmax(pred)
            rules.append(f"IF {rule} THEN class={pred}")
        else:
            # Left branch
            left_cond = f"X[{node.feature_idx}] <= {node.threshold:.4f}"
            self._extract_rules(node.left_child, conditions + [left_cond], rules)

            # Right branch
            right_cond = f"X[{node.feature_idx}] > {node.threshold:.4f}"
            self._extract_rules(node.right_child, conditions + [right_cond], rules)

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"TreeGenotype(depth={self.get_depth()}, "
            f"nodes={self.get_num_nodes()}, "
            f"leaves={self.get_num_leaves()}, "
            f"features={self.get_num_features_used()}/{self.n_features})"
        )


def create_leaf_node(prediction: Union[int, float, np.ndarray], depth: int = 0) -> Node:
    """Factory function to create a leaf node."""
    return Node(node_type="leaf", prediction=prediction, depth=depth)


def create_internal_node(
    feature_idx: int, threshold: float, left_child: Node, right_child: Node, depth: int = 0
) -> Node:
    """Factory function to create an internal node."""
    return Node(
        node_type="internal",
        feature_idx=feature_idx,
        threshold=threshold,
        operator="<=",
        left_child=left_child,
        right_child=right_child,
        depth=depth,
    )
