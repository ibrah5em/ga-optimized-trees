"""Fitness calculation for decision trees.

This module provides:
- TreePredictor: Makes predictions using a tree genotype (LDD-4: vectorized).
- InterpretabilityCalculator: Computes composite interpretability scores.
- FitnessCalculator: Combines accuracy and interpretability into a fitness value.

Changes from original:
- LDD-1:  Pareto mode raises clearly instead of silent fallback.
- LDD-3:  Optional validation set for generalization-aware fitness.
- LDD-4:  Vectorized batch prediction replaces per-sample Python loop.
- LDD-5:  Input validation on all public APIs.
- LDD-6:  Iterative prediction to avoid recursion depth issues.
- LDD-7:  Dynamic max_nodes derived from tree.max_depth.
- LDD-8:  Feature coherence returns 0.5 for leaf-only trees.
- LDD-11: Unreachable leaves inherit dataset prior, not arbitrary 0.
- LDD-12: Configurable classification metric (accuracy, f1, balanced_accuracy).
"""

from typing import Dict, Optional, Tuple, Union

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    mean_squared_error,
    r2_score,
)

# ---------------------------------------------------------------------------
# Named constants (LDD-7: no magic numbers)
# ---------------------------------------------------------------------------

#: Fallback maximum number of nodes when max_depth is unavailable.
#: Corresponds to a full binary tree of depth 6 (2^7 - 1 = 127).
DEFAULT_MAX_NODES_FALLBACK: int = 127

#: Hard ceiling on tree depth to prevent recursion issues (LDD-6).
MAX_ALLOWED_DEPTH: int = 50

#: Valid fitness modes.
VALID_MODES = frozenset({"weighted_sum", "pareto"})

#: Valid classification metrics for fitness evaluation.
VALID_CLASSIFICATION_METRICS = frozenset(
    {"accuracy", "f1_macro", "f1_weighted", "balanced_accuracy"}
)

#: Valid regression metrics for fitness evaluation.
VALID_REGRESSION_METRICS = frozenset({"neg_mse", "r2"})


class TreePredictor:
    """Make predictions with a tree genotype.

    Supports both classification and regression trees. Predictions
    are generated via iterative (stack-based) tree traversal to avoid
    recursion depth issues (LDD-6).
    """

    @staticmethod
    def predict(tree, X: np.ndarray) -> np.ndarray:
        """Predict labels for input data using vectorized batch traversal.

        Args:
            tree: TreeGenotype instance with a valid root node.
            X: Feature matrix of shape ``(n_samples, n_features)``.

        Returns:
            Predictions array of shape ``(n_samples,)``.

        Raises:
            ValueError: If *X* is not 2-D or has fewer features than
                the tree expects.
        """
        # --- LDD-5: input validation ---
        if X.ndim != 2:
            raise ValueError(f"X must be 2-D, got {X.ndim}-D array.")
        if X.shape[0] == 0:
            return np.array([], dtype=float)
        if X.shape[1] < tree.n_features:
            raise ValueError(
                f"X has {X.shape[1]} features but tree expects " f"at least {tree.n_features}."
            )

        # --- LDD-4: vectorized batch prediction ---
        result = TreePredictor._predict_batch(tree.root, X)
        if tree.task_type == "classification":
            return result.astype(int)
        return result

    @staticmethod
    def _predict_batch(root, X: np.ndarray) -> np.ndarray:
        """Vectorized prediction using iterative index-set partitioning.

        Instead of one recursive call per sample, we push
        ``(node, sample_indices)`` tuples onto a stack and process
        them with numpy boolean indexing.  This reduces Python call
        overhead from ``O(n_samples * depth)`` to ``O(n_nodes)`` calls
        with ``O(n)`` numpy operations each (LDD-4).
        """
        n_samples = X.shape[0]
        predictions = np.zeros(n_samples, dtype=float)
        # Stack of (node, indices_array)
        stack = [(root, np.arange(n_samples))]

        while stack:
            node, indices = stack.pop()

            if len(indices) == 0:
                continue

            if node.is_leaf():
                pred = node.prediction
                if isinstance(pred, np.ndarray):
                    pred = np.argmax(pred)
                if pred is None:
                    pred = 0.0
                predictions[indices] = pred
                continue

            # --- LDD-6: guard against malformed internal nodes ---
            if node.left_child is None or node.right_child is None:
                predictions[indices] = 0.0
                continue

            feature_vals = X[indices, node.feature_idx]
            left_mask = feature_vals <= node.threshold
            right_mask = ~left_mask

            left_indices = indices[left_mask]
            right_indices = indices[right_mask]

            if len(left_indices) > 0:
                stack.append((node.left_child, left_indices))
            if len(right_indices) > 0:
                stack.append((node.right_child, right_indices))

        return predictions

    @staticmethod
    def fit_leaf_predictions(tree, X: np.ndarray, y: np.ndarray) -> None:
        """Update leaf predictions based on data.

        Args:
            tree: TreeGenotype instance.
            X: Feature matrix of shape ``(n_samples, n_features)``.
            y: Target array of shape ``(n_samples,)``.

        Notes:
            LDD-11: Unreachable leaves receive the dataset-global prior
            (majority class for classification, global mean for regression)
            instead of an arbitrary ``0``.
        """
        if X.shape[0] == 0:
            return

        # --- LDD-11: compute dataset-global prior ---
        if tree.task_type == "classification":
            unique, counts = np.unique(y, return_counts=True)
            global_prior: Union[int, float] = int(unique[np.argmax(counts)])
        else:
            global_prior = float(np.mean(y))

        # Assign samples to leaves via iterative traversal (LDD-4 / LDD-6)
        leaf_samples: Dict[int, list] = {}
        stack = [(tree.root, np.arange(len(X)))]

        while stack:
            node, indices = stack.pop()
            if len(indices) == 0:
                continue
            if node.is_leaf():
                leaf_samples[node.node_id] = [y[i] for i in indices]
                continue
            if node.left_child is None or node.right_child is None:
                # Malformed internal node — treat as leaf
                leaf_samples.setdefault(node.node_id, [])
                continue

            feature_vals = X[indices, node.feature_idx]
            left_mask = feature_vals <= node.threshold
            left_idx = indices[left_mask]
            right_idx = indices[~left_mask]
            stack.append((node.left_child, left_idx))
            stack.append((node.right_child, right_idx))

        # Update predictions
        for node in tree.get_all_leaves():
            if node.node_id in leaf_samples and leaf_samples[node.node_id]:
                samples = leaf_samples[node.node_id]
                if tree.task_type == "classification":
                    u, c = np.unique(samples, return_counts=True)
                    node.prediction = int(u[np.argmax(c)])
                else:
                    node.prediction = float(np.mean(samples))
            else:
                # LDD-11: use global prior, not arbitrary 0
                node.prediction = global_prior


class InterpretabilityCalculator:
    """Calculate interpretability metrics for decision trees.

    Each sub-metric returns a value in ``[0, 1]`` where higher means
    more interpretable.
    """

    @staticmethod
    def calculate_composite_score(tree, weights: Dict[str, float]) -> float:
        """Calculate composite interpretability score.

        Args:
            tree: TreeGenotype instance.
            weights: Dictionary mapping metric names to their weights.
                Recognized keys: ``node_complexity``, ``feature_coherence``,
                ``tree_balance``, ``semantic_coherence``.

        Returns:
            Weighted score in ``[0, 1]`` (assuming sub-weights sum to 1).
        """
        score = 0.0

        if "node_complexity" in weights:
            score += weights["node_complexity"] * InterpretabilityCalculator._node_complexity(tree)

        if "feature_coherence" in weights:
            score += weights["feature_coherence"] * InterpretabilityCalculator._feature_coherence(
                tree
            )

        if "tree_balance" in weights:
            score += weights["tree_balance"] * tree.get_tree_balance()

        if "semantic_coherence" in weights:
            score += weights["semantic_coherence"] * InterpretabilityCalculator._semantic_coherence(
                tree
            )

        return score

    @staticmethod
    def _node_complexity(tree) -> float:
        """Node complexity metric — fewer nodes = more interpretable.

        Formula: ``M_1(T) = 1 - nodes(T) / max_nodes``

        LDD-7: ``max_nodes`` is dynamically derived from the tree's
        ``max_depth`` as ``2^(max_depth+1) - 1`` (full binary tree).
        """
        num_nodes = tree.get_num_nodes()

        # LDD-7: derive max_nodes from tree constraints
        if hasattr(tree, "max_depth") and tree.max_depth is not None and tree.max_depth > 0:
            max_nodes = (2 ** (tree.max_depth + 1)) - 1
        else:
            max_nodes = DEFAULT_MAX_NODES_FALLBACK

        return max(0.0, 1.0 - num_nodes / max_nodes)

    @staticmethod
    def _feature_coherence(tree) -> float:
        """Feature coherence — reward using fewer features.

        Formula: ``M_2(T) = 1 - unique_features(T) / total_features``

        LDD-8: A tree using *zero* features (leaf-only) returns 0.5
        instead of 1.0 to avoid rewarding trivial trees maximally.
        """
        features_used = tree.get_features_used()

        # LDD-8: leaf-only trees get neutral score, not maximum
        if not features_used:
            return 0.5

        total_features = tree.n_features
        if total_features == 0:
            return 0.5

        coherence = 1.0 - (len(features_used) / total_features)
        return max(0.0, coherence)

    @staticmethod
    def _semantic_coherence(tree) -> float:
        """Semantic coherence — consistency of feature depth positions.

        Trees that reuse the same features at consistent depths exhibit
        logical coherence.  Returns 1.0 for perfect consistency.
        """
        internal_nodes = tree.get_internal_nodes()
        if not internal_nodes:
            return 1.0

        feature_depths: Dict[int, list] = {}
        for node in internal_nodes:
            if node.feature_idx is not None:
                feature_depths.setdefault(node.feature_idx, []).append(node.depth)

        if not feature_depths:
            return 1.0

        max_depth = tree.get_depth()
        if max_depth == 0:
            return 1.0

        consistency_scores = []
        for depths in feature_depths.values():
            if len(depths) == 1:
                consistency_scores.append(1.0)
            else:
                depth_std = float(np.std(depths))
                score = 1.0 - min(depth_std / max_depth, 1.0)
                consistency_scores.append(score)

        return float(np.mean(consistency_scores))


class FitnessCalculator:
    """Main fitness calculator combining accuracy and interpretability.

    Args:
        mode: ``"weighted_sum"`` for scalar fitness, ``"pareto"`` for
            multi-objective (returns tuple).  LDD-1: ``"pareto"`` now
            genuinely returns a tuple and does not silently fall back
            to weighted sum.
        accuracy_weight: Weight for accuracy objective, in ``[0, 1]``.
        interpretability_weight: Weight for interpretability, in ``[0, 1]``.
        interpretability_weights: Sub-weights for interpretability components.
        classification_metric: Metric for classification fitness (LDD-12).
            One of ``"accuracy"``, ``"f1_macro"``, ``"f1_weighted"``,
            ``"balanced_accuracy"``.
        regression_metric: Metric for regression fitness.
            One of ``"neg_mse"`` (transformed to ``1/(1+MSE)``), ``"r2"``.
    """

    def __init__(
        self,
        mode: str = "weighted_sum",
        accuracy_weight: float = 0.7,
        interpretability_weight: float = 0.3,
        interpretability_weights: Optional[Dict[str, float]] = None,
        classification_metric: str = "accuracy",
        regression_metric: str = "neg_mse",
    ):
        # --- LDD-5: input validation ---
        if mode not in VALID_MODES:
            raise ValueError(f"mode must be one of {VALID_MODES}, got '{mode}'.")
        if not (0.0 <= accuracy_weight <= 1.0):
            raise ValueError(f"accuracy_weight must be in [0, 1], got {accuracy_weight}.")
        if not (0.0 <= interpretability_weight <= 1.0):
            raise ValueError(
                f"interpretability_weight must be in [0, 1], got {interpretability_weight}."
            )
        if classification_metric not in VALID_CLASSIFICATION_METRICS:
            raise ValueError(
                f"classification_metric must be one of "
                f"{VALID_CLASSIFICATION_METRICS}, got '{classification_metric}'."
            )
        if regression_metric not in VALID_REGRESSION_METRICS:
            raise ValueError(
                f"regression_metric must be one of "
                f"{VALID_REGRESSION_METRICS}, got '{regression_metric}'."
            )

        self.mode = mode
        self.accuracy_weight = accuracy_weight
        self.interpretability_weight = interpretability_weight
        self.classification_metric = classification_metric
        self.regression_metric = regression_metric

        if interpretability_weights is None:
            self.interpretability_weights: Dict[str, float] = {
                "node_complexity": 0.4,
                "feature_coherence": 0.3,
                "tree_balance": 0.2,
                "semantic_coherence": 0.1,
            }
        else:
            self.interpretability_weights = interpretability_weights

        self.predictor = TreePredictor()
        self.interp_calc = InterpretabilityCalculator()

    def calculate_fitness(
        self,
        tree,
        X: np.ndarray,
        y: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> Union[float, Tuple[float, float]]:
        """Calculate fitness for a tree.

        Args:
            tree: TreeGenotype instance.
            X: Training features — used to fit leaf predictions.
            y: Training labels.
            X_val: Optional validation features for generalization
                fitness (LDD-3).
            y_val: Optional validation labels.

        Returns:
            Scalar fitness (``weighted_sum`` mode) or
            ``(accuracy, interpretability)`` tuple (``pareto`` mode).

        Raises:
            ValueError: If inputs are invalid.
        """
        # --- LDD-5: validate inputs ---
        if X.ndim != 2:
            raise ValueError(f"X must be 2-D, got {X.ndim}-D.")
        if len(X) != len(y):
            raise ValueError(f"X and y length mismatch: {len(X)} vs {len(y)}.")

        # Fit leaf predictions on training data
        self.predictor.fit_leaf_predictions(tree, X, y)

        # --- LDD-3: evaluate on validation set if provided ---
        if X_val is not None and y_val is not None:
            X_eval, y_eval = X_val, y_val
        else:
            X_eval, y_eval = X, y

        y_pred = self.predictor.predict(tree, X_eval)

        # --- LDD-12: configurable classification metric ---
        if tree.task_type == "classification":
            accuracy = self._compute_classification_metric(y_eval, y_pred)
        else:
            accuracy = self._compute_regression_metric(y_eval, y_pred)

        # Calculate interpretability
        interpretability = self.interp_calc.calculate_composite_score(
            tree, self.interpretability_weights
        )

        # Store individual scores on the tree
        tree.accuracy_ = accuracy
        tree.interpretability_ = interpretability

        # --- LDD-1: Pareto mode returns tuple genuinely ---
        if self.mode == "pareto":
            return (accuracy, interpretability)

        # weighted_sum mode
        fitness: float = (
            self.accuracy_weight * accuracy + self.interpretability_weight * interpretability
        )
        return fitness

    def _compute_classification_metric(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute the configured classification metric (LDD-12)."""
        if self.classification_metric == "accuracy":
            return float(accuracy_score(y_true, y_pred))
        elif self.classification_metric == "f1_macro":
            return float(f1_score(y_true, y_pred, average="macro", zero_division=0))
        elif self.classification_metric == "f1_weighted":
            return float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
        elif self.classification_metric == "balanced_accuracy":
            return float(balanced_accuracy_score(y_true, y_pred))
        else:
            return float(accuracy_score(y_true, y_pred))

    def _compute_regression_metric(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute the configured regression metric."""
        if self.regression_metric == "r2":
            return float(max(0.0, r2_score(y_true, y_pred)))
        else:  # neg_mse
            mse = mean_squared_error(y_true, y_pred)
            return 1.0 / (1.0 + mse)
