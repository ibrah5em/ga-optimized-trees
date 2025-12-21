"""
Decision Tree Genotype Representation

This module defines the core tree structure used by the genetic algorithm.
Trees are represented as binary decision trees with constrained structure.
"""

import copy
from dataclasses import dataclass, field
from ga_trees.configs.bayesian_config import BayesianConfig
from typing import List, Literal, Optional, Tuple, Union, Dict, Any

import numpy as np
import math


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
    # Bayesian leaf parameters: Dirichlet concentration for class probabilities
    leaf_alpha: Optional[np.ndarray] = None
    # Number of training samples seen at this leaf (for Bayesian updating)
    leaf_samples_count: int = 0

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
        new = copy.deepcopy(self)
        # Ensure leaf_alpha is a numpy array copy if present
        if getattr(new, "leaf_alpha", None) is not None:
            try:
                new.leaf_alpha = np.array(new.leaf_alpha, dtype=float, copy=True)
            except Exception:
                new.leaf_alpha = np.array(new.leaf_alpha, dtype=float)
        return new


@dataclass
class BayesianNode(Node):
    """An internal node with Bayesian/probabilistic threshold parameters.

    Backward compatible: `threshold` is set to `threshold_mean` when available
    so existing deterministic code continues to work.
    """

    # Distribution parameters for the split threshold
    threshold_mean: Optional[float] = None
    threshold_std: Optional[float] = None
    threshold_dist_type: Literal["normal", "laplace", "uniform"] = "normal"

    def __post_init__(self):
        # Ensure base dataclass post init semantics: propagate threshold_mean
        if self.threshold is None and self.threshold_mean is not None:
            self.threshold = self.threshold_mean

    def copy(self) -> "BayesianNode":
        """Deep-copy a BayesianNode, preserving numpy arrays and scalar types."""
        new = copy.deepcopy(self)
        if getattr(new, "leaf_alpha", None) is not None:
            try:
                new.leaf_alpha = np.array(new.leaf_alpha, dtype=float, copy=True)
            except Exception:
                new.leaf_alpha = np.array(new.leaf_alpha, dtype=float)
        if getattr(new, "threshold_mean", None) is not None:
            new.threshold_mean = float(new.threshold_mean)
        if getattr(new, "threshold_std", None) is not None:
            new.threshold_std = float(new.threshold_std)
        return new


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

    # Operation mode: 'deterministic' (legacy) or 'bayesian' (probabilistic)
    mode: Literal["deterministic", "bayesian"] = "deterministic"

    # Bayesian-specific configuration/hyperparameters
    bayesian_config: Dict[str, Any] = field(default_factory=dict)

    # Constraints
    max_depth: int = 5
    min_samples_split: int = 10
    min_samples_leaf: int = 5
    max_features: Optional[int] = None

    # Metadata
    fitness_: Optional[float] = None
    accuracy_: Optional[float] = None
    interpretability_: Optional[float] = None
    # Bayesian evaluation metrics
    mean_calibration_error: Optional[float] = None
    prediction_uncertainty: Optional[float] = None

    def __post_init__(self):
        """Initialize derived attributes."""
        if self.max_features is None:
            self.max_features = self.n_features
        # Normalize bayesian_config to a BayesianConfig instance when provided
        if isinstance(self.bayesian_config, dict):
            try:
                self.bayesian_config = BayesianConfig.from_dict(self.bayesian_config)
            except Exception:
                # leave as dict on failure
                pass
        elif self.bayesian_config is None:
            self.bayesian_config = BayesianConfig()

        self._assign_node_ids(self.root, 0)

    def is_bayesian(self) -> bool:
        """Return True if tree is operating in bayesian mode."""
        return self.mode == "bayesian"

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
            # Accept either deterministic threshold or Bayesian threshold_mean
            has_threshold = getattr(node, "threshold", None) is not None
            has_bayes_mean = getattr(node, "threshold_mean", None) is not None
            if not (has_threshold or has_bayes_mean):
                errors.append(f"Internal node {node.node_id} missing threshold")

        # Check leaf nodes have predictions or Bayesian alpha vectors
        for node in self.get_all_leaves():
            has_pred = getattr(node, "prediction", None) is not None
            has_alpha = getattr(node, "leaf_alpha", None) is not None
            if not (has_pred or has_alpha):
                errors.append(f"Leaf node {node.node_id} missing prediction or leaf_alpha")
            if has_alpha:
                try:
                    if len(node.leaf_alpha) != self.n_classes:
                        errors.append(
                            f"Leaf node {node.node_id} leaf_alpha length mismatch: expected {self.n_classes}, got {len(node.leaf_alpha)}"
                        )
                except Exception:
                    errors.append(f"Leaf node {node.node_id} has invalid leaf_alpha")
                if getattr(node, "leaf_samples_count", 0) < 0:
                    errors.append(f"Leaf node {node.node_id} has negative leaf_samples_count")

        # Check tree structure
        if not self._check_structure(self.root):
            errors.append("Tree structure is inconsistent")

        # If running in bayesian mode, run additional bayesian checks
        if self.mode == "bayesian":
            ok_b, errs_b = self.validate_bayesian()
            if not ok_b:
                errors.extend(errs_b)

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
        new = copy.deepcopy(self)
        # Ensure bayesian_config is copied as a plain dict
        try:
            new.bayesian_config = dict(self.bayesian_config) if self.bayesian_config is not None else {}
        except Exception:
            new.bayesian_config = copy.deepcopy(self.bayesian_config)
        return new

    def validate_bayesian(self) -> Tuple[bool, List[str]]:
        """Validate Bayesian-specific structural and parameter constraints.

        Returns (is_valid, errors).
        """
        errors: List[str] = []

        # Internal nodes: ensure Bayesian params present for BayesianNode
        for node in self.get_internal_nodes():
            if isinstance(node, BayesianNode):
                if getattr(node, "threshold_mean", None) is None:
                    errors.append(f"Bayesian internal node {node.node_id} missing threshold_mean")
                if getattr(node, "threshold_std", None) is None:
                    errors.append(f"Bayesian internal node {node.node_id} missing threshold_std")
                else:
                    try:
                        if float(node.threshold_std) < 0:
                            errors.append(f"Bayesian internal node {node.node_id} has negative threshold_std")
                    except Exception:
                        errors.append(f"Bayesian internal node {node.node_id} invalid threshold_std")
                if getattr(node, "threshold_dist_type", None) not in ("normal", "laplace", "uniform"):
                    errors.append(f"Bayesian internal node {node.node_id} invalid threshold_dist_type")

        # Leaf nodes: ensure leaf_alpha matches n_classes
        for node in self.get_all_leaves():
            if getattr(node, "leaf_alpha", None) is not None:
                try:
                    alpha = np.array(node.leaf_alpha, dtype=float)
                    if alpha.size != self.n_classes:
                        errors.append(
                            f"Bayesian leaf node {node.node_id} leaf_alpha length mismatch: expected {self.n_classes}, got {alpha.size}"
                        )
                    if np.any(alpha < 0):
                        errors.append(f"Bayesian leaf node {node.node_id} has negative entries in leaf_alpha")
                except Exception:
                    errors.append(f"Bayesian leaf node {node.node_id} has invalid leaf_alpha")
                if getattr(node, "leaf_samples_count", 0) < 0:
                    errors.append(f"Bayesian leaf node {node.node_id} has negative leaf_samples_count")

        return (len(errors) == 0, errors)

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
                "mode": self.mode,
                "bayesian_config": self.bayesian_config,
                "mean_calibration_error": self.mean_calibration_error,
                "prediction_uncertainty": self.prediction_uncertainty,
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
            # If BayesianNode, include distribution parameters
            if isinstance(node, BayesianNode):
                result.update(
                    {
                        "threshold_mean": float(node.threshold_mean) if node.threshold_mean is not None else None,
                        "threshold_std": float(node.threshold_std) if node.threshold_std is not None else None,
                        "threshold_dist_type": node.threshold_dist_type,
                    }
                )
        else:
            if getattr(node, "leaf_alpha", None) is not None:
                # Prefer representing Bayesian leaf posterior params
                result["leaf_alpha"] = (
                    node.leaf_alpha.tolist() if isinstance(node.leaf_alpha, np.ndarray) else list(node.leaf_alpha)
                )
                result["leaf_samples_count"] = int(node.leaf_samples_count)
                # Also include prediction if present (e.g., MAP)
                if node.prediction is not None:
                    result["prediction"] = node.prediction.tolist() if isinstance(node.prediction, np.ndarray) else node.prediction
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
    return Node(node_type="leaf", prediction=prediction, leaf_alpha=None, leaf_samples_count=0, depth=depth)


def create_bayesian_leaf_node(
    leaf_alpha: Union[List[float], np.ndarray], leaf_samples_count: int = 0, depth: int = 0
) -> Node:
    """Factory to create a Bayesian leaf node storing Dirichlet concentration params.

    The `prediction` can be left None; downstream prediction logic will use the
    posterior mean alpha / sum(alpha) if needed.
    """
    arr = np.array(leaf_alpha, dtype=float)
    return Node(node_type="leaf", prediction=None, leaf_alpha=arr, leaf_samples_count=int(leaf_samples_count), depth=depth)


def sample_threshold(node: Node, rng: Optional[np.random.Generator] = None) -> Optional[float]:
    """Sample a threshold value for a (possibly Bayesian) internal node.

    - If `node` is a `BayesianNode` with distribution params, sample according
      to the specified family (normal, laplace, uniform).
    - If no distribution is available, returns the deterministic `threshold`.
    """
    if rng is None:
        rng = np.random.default_rng()

    mean = getattr(node, "threshold_mean", None)
    std = getattr(node, "threshold_std", None)
    dist = getattr(node, "threshold_dist_type", None)

    # Deterministic fallback
    if mean is None or std is None or std == 0 or dist is None:
        return getattr(node, "threshold", None)

    if dist == "normal":
        return float(rng.normal(loc=mean, scale=std))
    if dist == "laplace":
        # Laplace scale parameter b where std = sqrt(2)*b
        b = float(std) / math.sqrt(2.0)
        return float(rng.laplace(loc=mean, scale=b))
    if dist == "uniform":
        # Derive half-width from std: std = (high-low)/sqrt(12) => halfwidth = sqrt(3)*std
        half = float(std) * math.sqrt(3.0)
        return float(rng.uniform(low=mean - half, high=mean + half))

    # Unknown distribution, fallback to deterministic threshold
    return getattr(node, "threshold", None)


def sample_leaf_distribution(node: Node, n_classes: int, rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """Return a sampled class probability vector for a leaf.

    - If the leaf has `leaf_alpha`, sample from Dirichlet(leaf_alpha).
    - If the leaf has a deterministic `prediction`, return one-hot (or normalized
      probability vector if `prediction` is a vector).
    """
    if rng is None:
        rng = np.random.default_rng()

    if getattr(node, "leaf_alpha", None) is not None:
        alpha = np.array(node.leaf_alpha, dtype=float)
        if alpha.size != n_classes:
            raise ValueError("leaf_alpha length does not match n_classes")
        return rng.dirichlet(alpha)

    # Deterministic prediction fallback
    pred = getattr(node, "prediction", None)
    if pred is None:
        # Unknown: return uniform
        return np.ones(n_classes, dtype=float) / float(n_classes)

    if isinstance(pred, np.ndarray):
        vec = np.array(pred, dtype=float)
        s = vec.sum()
        if s == 0:
            return np.ones(n_classes, dtype=float) / float(n_classes)
        return vec / float(s)

    # Scalar predicted class -> one-hot
    vec = np.zeros(n_classes, dtype=float)
    idx = int(pred)
    if idx < 0 or idx >= n_classes:
        return np.ones(n_classes, dtype=float) / float(n_classes)
    vec[idx] = 1.0
    return vec


def get_soft_decision_prob(node: Node, x_row: np.ndarray, mc_samples: int = 200, rng: Optional[np.random.Generator] = None) -> float:
    """Compute probability of routing left (X[feature_idx] <= threshold).

    Uses analytic expressions for common families (normal, laplace, uniform)
    when the node exposes `threshold_mean` and `threshold_std`. Falls back to
    Monte Carlo by sampling thresholds when analytic is not available.
    Returns a float in [0, 1].
    """
    if node is None or node.is_leaf():
        return 0.0
    if rng is None:
        rng = np.random.default_rng()

    x_val = float(x_row[node.feature_idx])

    mean = getattr(node, "threshold_mean", None)
    std = getattr(node, "threshold_std", None)
    dist = getattr(node, "threshold_dist_type", None)

    # Deterministic threshold
    if mean is None or std is None or std == 0 or dist is None:
        thr = getattr(node, "threshold", None)
        if thr is None:
            return 0.0
        return 1.0 if x_val <= thr else 0.0

    if dist == "normal":
        # P(threshold >= x_val) = 1 - Phi((x - mu)/sigma)
        z = (x_val - mean) / float(std)
        phi = 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
        return float(max(0.0, min(1.0, 1.0 - phi)))

    if dist == "laplace":
        b = float(std) / math.sqrt(2.0)
        # CDF of Laplace at x
        if x_val < mean:
            cdf = 0.5 * math.exp((x_val - mean) / b)
        else:
            cdf = 1.0 - 0.5 * math.exp(-(x_val - mean) / b)
        return float(max(0.0, min(1.0, 1.0 - cdf)))

    if dist == "uniform":
        half = float(std) * math.sqrt(3.0)
        low = mean - half
        high = mean + half
        if x_val < low:
            cdf = 0.0
        elif x_val > high:
            cdf = 1.0
        else:
            cdf = (x_val - low) / (high - low)
        return float(max(0.0, min(1.0, 1.0 - cdf)))

    # Unknown distribution: Monte Carlo estimate
    samples = 0
    for _ in range(mc_samples):
        thr = sample_threshold(node, rng=rng)
        if thr is None:
            continue
        if x_val <= thr:
            samples += 1
    return float(samples) / float(mc_samples)


def create_internal_node(
    feature_idx: int, threshold: float, left_child: Node, right_child: Node, depth: int = 0
) -> Node:
    """Factory function to create an internal node."""
    # set child depths for structural consistency
    if left_child is not None:
        left_child.depth = depth + 1
    if right_child is not None:
        right_child.depth = depth + 1
    return Node(
        node_type="internal",
        feature_idx=feature_idx,
        threshold=threshold,
        operator="<=",
        left_child=left_child,
        right_child=right_child,
        depth=depth,
    )


def create_bayesian_internal_node(
    feature_idx: int,
    threshold_mean: float,
    threshold_std: float,
    threshold_dist_type: Literal["normal", "laplace", "uniform"],
    left_child: Node = None,
    right_child: Node = None,
    depth: int = 0,
) -> BayesianNode:
    """Factory function to create a Bayesian internal node.

    This sets the deterministic `threshold` to `threshold_mean` for
    backward compatibility while storing distribution params.
    """
    # set child depths for structural consistency (same behavior as create_internal_node)
    if left_child is not None:
        left_child.depth = depth + 1
    if right_child is not None:
        right_child.depth = depth + 1
    return BayesianNode(
        node_type="internal",
        feature_idx=feature_idx,
        threshold=threshold_mean,
        threshold_mean=threshold_mean,
        threshold_std=threshold_std,
        threshold_dist_type=threshold_dist_type,
        operator="<=",
        left_child=left_child,
        right_child=right_child,
        depth=depth,
    )
