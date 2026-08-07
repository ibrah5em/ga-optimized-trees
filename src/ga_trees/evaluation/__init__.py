"""Evaluation package exports.

Expose evaluation and analysis classes for easy import.
"""

from .explainability import TreeExplainer
from .feature_importance import FeatureImportanceAnalyzer
from .metrics import MetricsCalculator
from .statistics import (
    compare_across_datasets,
    compare_all_to_reference,
    equivalence_test,
    friedman_nemenyi,
    holm_adjust,
    per_dataset_means,
    summarize,
)
from .tree_visualizer import TreeVisualizer

__all__ = [
    "MetricsCalculator",
    "FeatureImportanceAnalyzer",
    "TreeVisualizer",
    "TreeExplainer",
    "summarize",
    "holm_adjust",
    "compare_across_datasets",
    "compare_all_to_reference",
    "equivalence_test",
    "friedman_nemenyi",
    "per_dataset_means",
]
