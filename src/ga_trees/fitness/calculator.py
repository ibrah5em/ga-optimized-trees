"""Fitness calculation for decision trees."""

from typing import Dict, Tuple

import numpy as np
from scipy.stats import entropy
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error


class TreePredictor:
    """Make predictions with a tree genotype."""

    @staticmethod
    def predict(tree, X: np.ndarray) -> np.ndarray:
        """Predict labels for input data."""
        predictions = np.zeros(len(X))
        for i, x in enumerate(X):
            predictions[i] = TreePredictor._predict_single(tree.root, x)
        return predictions

    @staticmethod
    def _predict_single(node, x: np.ndarray):
        """Predict single instance."""
        if node.is_leaf():
            if isinstance(node.prediction, np.ndarray):
                return np.argmax(node.prediction)
            return node.prediction

        # Traverse tree
        if x[node.feature_idx] <= node.threshold:
            return TreePredictor._predict_single(node.left_child, x)
        else:
            return TreePredictor._predict_single(node.right_child, x)

    @staticmethod
    def fit_leaf_predictions(tree, X: np.ndarray, y: np.ndarray):
        """Update leaf predictions based on data."""
        # Assign samples to leaves
        leaf_samples = {}
        for i, x in enumerate(X):
            leaf = TreePredictor._find_leaf(tree.root, x)
            if leaf.node_id not in leaf_samples:
                leaf_samples[leaf.node_id] = []
            leaf_samples[leaf.node_id].append(y[i])

        # Update predictions
        for node in tree.get_all_leaves():
            if node.node_id in leaf_samples:
                samples = leaf_samples[node.node_id]
                if tree.task_type == "classification":
                    # Most common class
                    unique, counts = np.unique(samples, return_counts=True)
                    node.prediction = int(unique[np.argmax(counts)])
                else:
                    # Mean for regression
                    node.prediction = float(np.mean(samples))
            else:
                # No samples reached this leaf
                node.prediction = 0

    @staticmethod
    def _find_leaf(node, x: np.ndarray):
        """Find leaf node for instance."""
        if node.is_leaf():
            return node
        if x[node.feature_idx] <= node.threshold:
            return TreePredictor._find_leaf(node.left_child, x)
        else:
            return TreePredictor._find_leaf(node.right_child, x)


class InterpretabilityCalculator:
    """Calculate interpretability metrics."""

    @staticmethod
    def calculate_composite_score(tree, weights: Dict[str, float]) -> float:
        """
        Calculate composite interpretability score.

        Args:
            tree: TreeGenotype
            weights: Dictionary with keys:
                - node_complexity
                - feature_coherence
                - tree_balance
                - semantic_coherence

        Returns:
            Score in [0, 1] where higher is more interpretable
        """
        score = 0.0

        # Node complexity (fewer nodes = more interpretable)
        if "node_complexity" in weights:
            node_score = InterpretabilityCalculator._node_complexity(tree)
            score += weights["node_complexity"] * node_score

        # Feature coherence (related features used together)
        if "feature_coherence" in weights:
            coherence = InterpretabilityCalculator._feature_coherence(tree)
            score += weights["feature_coherence"] * coherence

        # Tree balance (balanced trees easier to understand)
        if "tree_balance" in weights:
            balance = tree.get_tree_balance()
            score += weights["tree_balance"] * balance

        # Semantic coherence (consistent predictions in subtrees)
        if "semantic_coherence" in weights:
            semantic = InterpretabilityCalculator._semantic_coherence(tree)
            score += weights["semantic_coherence"] * semantic

        return score

    @staticmethod
    def _node_complexity(tree) -> float:
        """Node complexity metric (fewer nodes better)."""
        num_nodes = tree.get_num_nodes()
        # Normalize: assume max reasonable tree has 127 nodes (depth 7 full binary tree)
        max_nodes = 127
        return 1.0 / (1.0 + num_nodes / max_nodes)

    @staticmethod
    def _feature_coherence(tree) -> float:
        """Feature coherence - how well features are grouped."""
        # Simple version: ratio of unique features to total internal nodes
        internal_nodes = tree.get_internal_nodes()
        if not internal_nodes:
            return 1.0

        features_used = tree.get_features_used()
        if not features_used:
            return 1.0

        # Fewer unique features relative to nodes = more coherent
        coherence = 1.0 - (len(features_used) / len(internal_nodes))
        return max(0.0, coherence)

    @staticmethod
    def _semantic_coherence(tree) -> float:
        """Semantic coherence - consistency of predictions."""
        leaves = tree.get_all_leaves()
        if len(leaves) <= 1:
            return 1.0

        # Calculate entropy of leaf predictions
        predictions = [l.prediction for l in leaves if l.prediction is not None]
        if not predictions:
            return 0.5

        # For classification: lower entropy = more coherent
        try:
            unique, counts = np.unique(predictions, return_counts=True)
            if len(unique) == 1:
                return 1.0
            probs = counts / len(predictions)
            ent = entropy(probs)
            max_ent = np.log(len(unique))
            return 1.0 - (ent / max_ent if max_ent > 0 else 0)
        except:
            return 0.5


class FitnessCalculator:
    """Main fitness calculator."""

    def __init__(
        self,
        mode: str = "weighted_sum",
        accuracy_weight: float = 0.7,
        interpretability_weight: float = 0.3,
        interpretability_weights: Dict[str, float] = None,
    ):
        """
        Initialize fitness calculator.

        Args:
            mode: 'weighted_sum' or 'pareto'
            accuracy_weight: Weight for accuracy (0-1)
            interpretability_weight: Weight for interpretability (0-1)
            interpretability_weights: Sub-weights for interpretability components
        """
        self.mode = mode
        self.accuracy_weight = accuracy_weight
        self.interpretability_weight = interpretability_weight

        if interpretability_weights is None:
            self.interpretability_weights = {
                "node_complexity": 0.4,
                "feature_coherence": 0.3,
                "tree_balance": 0.2,
                "semantic_coherence": 0.1,
            }
        else:
            self.interpretability_weights = interpretability_weights

        self.predictor = TreePredictor()
        self.interp_calc = InterpretabilityCalculator()

    def calculate_fitness(self, tree, X: np.ndarray, y: np.ndarray) -> float:
        """
        Calculate fitness for a tree.

        Args:
            tree: TreeGenotype
            X: Features
            y: Labels

        Returns:
            Fitness score
        """
        # Fit leaf predictions to data
        self.predictor.fit_leaf_predictions(tree, X, y)

        # Calculate accuracy
        y_pred = self.predictor.predict(tree, X)

        if tree.task_type == "classification":
            accuracy = accuracy_score(y, y_pred)
        else:
            # For regression, convert MSE to accuracy-like metric
            mse = mean_squared_error(y, y_pred)
            accuracy = 1.0 / (1.0 + mse)

        # Calculate interpretability
        interpretability = self.interp_calc.calculate_composite_score(
            tree, self.interpretability_weights
        )

        # Store individual scores
        tree.accuracy_ = accuracy
        tree.interpretability_ = interpretability

        # Calculate fitness based on mode
        if self.mode == "weighted_sum":
            fitness = (
                self.accuracy_weight * accuracy + self.interpretability_weight * interpretability
            )
        else:
            # For Pareto mode, return tuple (accuracy, interpretability)
            # But GA engine expects scalar, so we return weighted sum anyway
            fitness = (
                self.accuracy_weight * accuracy + self.interpretability_weight * interpretability
            )

        return fitness
