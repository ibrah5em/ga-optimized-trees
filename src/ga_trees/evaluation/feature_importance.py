"""Feature importance analysis for evolved trees."""

from collections import Counter
from typing import Dict, List

import matplotlib.pyplot as plt


class FeatureImportanceAnalyzer:
    """Analyze feature importance in GA-evolved trees."""

    @staticmethod
    def calculate_feature_frequency(tree) -> Dict[int, int]:
        """Count how often each feature is used."""
        features_used = []

        def traverse(node):
            if node.is_internal():
                features_used.append(node.feature_idx)
                if node.left_child:
                    traverse(node.left_child)
                if node.right_child:
                    traverse(node.right_child)

        traverse(tree.root)
        return dict(Counter(features_used))

    @staticmethod
    def calculate_feature_depth_importance(tree) -> Dict[int, float]:
        """Calculate importance based on depth (higher = less important)."""
        importance = {}

        def traverse(node):
            if node.is_internal():
                # Importance = 1 / (depth + 1)
                weight = 1.0 / (node.depth + 1)
                importance[node.feature_idx] = importance.get(node.feature_idx, 0) + weight
                if node.left_child:
                    traverse(node.left_child)
                if node.right_child:
                    traverse(node.right_child)

        traverse(tree.root)

        # Normalize
        total = sum(importance.values())
        if total > 0:
            importance = {k: v / total for k, v in importance.items()}

        return importance

    @staticmethod
    def plot_feature_importance(
        importance: Dict[int, float], feature_names: List[str] = None, save_path: str = None
    ):
        """Visualize feature importance."""
        if not importance:
            print("No features used in tree!")
            return

        features = list(importance.keys())
        scores = list(importance.values())

        if feature_names:
            labels = [feature_names[f] for f in features]
        else:
            labels = [f"Feature {f}" for f in features]

        # Sort by importance
        sorted_pairs = sorted(zip(scores, labels), reverse=True)
        scores, labels = zip(*sorted_pairs)

        plt.figure(figsize=(10, 6))
        plt.barh(range(len(labels)), scores, color="skyblue", edgecolor="black")
        plt.yticks(range(len(labels)), labels)
        plt.xlabel("Importance Score", fontsize=12)
        plt.title("Feature Importance in GA Tree", fontsize=14, fontweight="bold")
        plt.grid(axis="x", alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.show()
