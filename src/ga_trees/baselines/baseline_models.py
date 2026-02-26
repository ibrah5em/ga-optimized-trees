"""Baseline model implementations for comparison."""

import logging
import warnings
from typing import Any, Dict

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

logger = logging.getLogger(__name__)


class BaselineModel:
    """Base class for baseline models."""

    def __init__(self, name: str, random_state: int = 42):
        self.name = name
        self.random_state = random_state
        self.model = None
        self.metrics_ = {}

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Train model."""
        raise NotImplementedError

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        return self.model.predict(X)

    def get_metrics(self) -> Dict[str, Any]:
        """Get model metrics."""
        return {
            "name": self.name,
            "depth": self.get_depth(),
            "num_nodes": self.get_num_nodes(),
            "num_leaves": self.get_num_leaves(),
            "features_used": self.get_num_features_used(),
        }

    def get_depth(self) -> int:
        """Get tree depth."""
        if hasattr(self.model, "tree_"):
            return self.model.tree_.max_depth
        return -1

    def get_num_nodes(self) -> int:
        """Get number of nodes."""
        if hasattr(self.model, "tree_"):
            return self.model.tree_.node_count
        return -1

    def get_num_leaves(self) -> int:
        """Get number of leaves."""
        if hasattr(self.model, "tree_"):
            return np.sum(self.model.tree_.children_left == -1)
        return -1

    def get_num_features_used(self) -> int:
        """Get number of features used."""
        if hasattr(self.model, "tree_"):
            features = self.model.tree_.feature
            return len(np.unique(features[features >= 0]))
        return -1


class CARTBaseline(BaselineModel):
    """Standard CART decision tree."""

    def __init__(
        self,
        max_depth: int = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        random_state: int = 42,
    ):
        super().__init__("CART", random_state)
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Train CART tree."""
        self.model = DecisionTreeClassifier(
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            random_state=self.random_state,
        )
        self.model.fit(X, y)
        return self


class PrunedCARTBaseline(BaselineModel):
    """CART with cost-complexity pruning."""

    def __init__(self, max_depth: int = None, random_state: int = 42):
        super().__init__("Pruned CART", random_state)
        self.max_depth = max_depth

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Train and prune CART tree."""
        # First train full tree
        tree = DecisionTreeClassifier(max_depth=self.max_depth, random_state=self.random_state)
        tree.fit(X, y)

        # Get pruning path
        path = tree.cost_complexity_pruning_path(X, y)
        ccp_alphas = path.ccp_alphas

        # Find best alpha via cross-validation (simplified)
        if len(ccp_alphas) > 1:
            best_alpha = ccp_alphas[len(ccp_alphas) // 2]
        else:
            best_alpha = 0.0

        # Train with optimal alpha
        self.model = DecisionTreeClassifier(ccp_alpha=best_alpha, random_state=self.random_state)
        self.model.fit(X, y)
        return self


class RandomForestBaseline(BaselineModel):
    """Random Forest ensemble."""

    def __init__(self, n_estimators: int = 100, max_depth: int = None, random_state: int = 42):
        super().__init__("Random Forest", random_state)
        self.n_estimators = n_estimators
        self.max_depth = max_depth

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Train random forest."""
        self.model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=self.random_state,
            n_jobs=-1,
        )
        self.model.fit(X, y)
        return self

    def get_metrics(self) -> Dict[str, Any]:
        """Get ensemble metrics."""
        metrics = super().get_metrics()
        # Average across trees
        metrics["depth"] = int(np.mean([tree.tree_.max_depth for tree in self.model.estimators_]))
        metrics["num_nodes"] = int(
            np.mean([tree.tree_.node_count for tree in self.model.estimators_])
        )
        return metrics


class XGBoostBaseline(BaselineModel):
    """XGBoost baseline (if available)."""

    def __init__(self, max_depth: int = 6, n_estimators: int = 100, random_state: int = 42):
        super().__init__("XGBoost", random_state)
        self.max_depth = max_depth
        self.n_estimators = n_estimators

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Train XGBoost."""
        try:
            from xgboost import XGBClassifier

            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)
                self.model = XGBClassifier(
                    max_depth=self.max_depth,
                    n_estimators=self.n_estimators,
                    random_state=self.random_state,
                    verbosity=0,
                )
                self.model.fit(X, y)
        except Exception as e:
            logger.warning("XGBoost unavailable, skipping: %s", e)
            self.model = None
        return self
