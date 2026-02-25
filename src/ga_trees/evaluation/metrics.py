"""Comprehensive metrics for model evaluation."""

from typing import Any, Dict

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


class MetricsCalculator:
    """Calculate comprehensive metrics."""

    @staticmethod
    def calculate_classification_metrics(
        y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray = None
    ) -> Dict[str, Any]:
        """Calculate classification metrics."""
        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
            "precision_weighted": precision_score(
                y_true, y_pred, average="weighted", zero_division=0
            ),
            "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
            "recall_weighted": recall_score(y_true, y_pred, average="weighted", zero_division=0),
            "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
            "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
            "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        }

        # ROC-AUC if probabilities available
        if y_prob is not None and len(np.unique(y_true)) == 2:
            try:
                metrics["roc_auc"] = roc_auc_score(y_true, y_prob[:, 1])
            except (ValueError, IndexError):
                # LDD-10: only catch expected failures (e.g., single class, wrong shape)
                pass

        return metrics

    @staticmethod
    def calculate_interpretability_metrics(tree) -> Dict[str, float]:
        """Calculate interpretability metrics."""
        return {
            "depth": tree.get_depth(),
            "num_nodes": tree.get_num_nodes(),
            "num_leaves": tree.get_num_leaves(),
            "features_used": tree.get_num_features_used(),
            "tree_balance": tree.get_tree_balance(),
            "interpretability_score": tree.interpretability_,
        }

    @staticmethod
    def print_classification_report(
        y_true: np.ndarray, y_pred: np.ndarray, target_names: list = None
    ):
        """Print detailed classification report."""
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred, target_names=target_names))

        print("\nConfusion Matrix:")
        cm = confusion_matrix(y_true, y_pred)
        print(cm)
