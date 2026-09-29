"""GOSDT regularisation path as a frontier method (Phase 4, exploratory).

``paper/PLAN.md`` Phase 4 asks for GOSDT (Lin et al. 2020) "as a baseline where
feasible". This is **not** part of the pre-registered protocol: it was added
after K1/K2 were known and is reported as an exploratory comparison only.

GOSDT solves for the provably optimal sparse tree over *binary* features under a
per-leaf penalty ``regularization``. Following McTavish et al. (2022), numeric
features are binarised with thresholds guessed from a gradient-boosted stump
ensemble fitted on the training fold only. Sweeping the penalty yields one
optimal tree per value — a frontier comparable to CART's cost-complexity path.

Requires the optional ``gosdt`` package.
"""

import json
from typing import List, Sequence, Tuple

import numpy as np

from ga_trees.benchmark.frontiers import FrontierMethod

#: Per-leaf penalties swept. The smallest sits near 1/n for the CC-18 sizes used
#: here; GOSDT warns (and slows) below that.
DEFAULT_REGULARIZATIONS: Tuple[float, ...] = (
    0.002,
    0.004,
    0.006,
    0.01,
    0.015,
    0.02,
    0.03,
    0.05,
    0.08,
    0.12,
)


def _patch_sklearn_compat() -> None:
    """gosdt 1.0.4 passes ``force_all_finite``, which scikit-learn 1.8 removed."""
    import gosdt._classifier as module
    from sklearn.utils.validation import check_array

    if getattr(module, "_ga_trees_patched", False):
        return

    def compatible(array, *args, force_all_finite=None, **kwargs):
        if force_all_finite is not None and "ensure_all_finite" not in kwargs:
            kwargs["ensure_all_finite"] = force_all_finite
        return check_array(array, *args, **kwargs)

    module.check_array = compatible
    module._ga_trees_patched = True


def _count_nodes(tree_json: dict) -> int:
    if "prediction" in tree_json:
        return 1
    return 1 + _count_nodes(tree_json["true"]) + _count_nodes(tree_json["false"])


class _NodeCount:
    def __init__(self, node_count: int):
        self.node_count = node_count


class _BinarisedGOSDT:
    """A fitted GOSDT tree behind its training-fold binariser.

    Exposes ``predict`` on raw features and ``tree_.node_count`` so that
    ``frontiers._score_candidates`` scores it exactly like a sklearn tree.
    """

    def __init__(self, binarizer, model):
        self.binarizer = binarizer
        self.model = model
        self.tree_ = _NodeCount(_count_nodes(json.loads(model.result_.model)[0]))

    def predict(self, X: np.ndarray) -> np.ndarray:
        import pandas as pd

        return self.model.predict(self.binarizer.transform(pd.DataFrame(X)))


class GOSDTPathFrontier(FrontierMethod):
    """One optimal sparse tree per regularisation value.

    Parameters
    ----------
    depth_budget : int
        GOSDT's depth budget, which counts the root as depth 1; ``max_depth + 1``
        gives the same limit as the other methods.
    time_limit : int
        Seconds per fit. A fit that times out returns GOSDT's best tree so far,
        which is no longer certified optimal; ``n_timeouts`` counts them.
    """

    name = "GOSDT (reg path)"

    def __init__(
        self,
        max_depth: int = 6,
        regularizations: Sequence[float] = DEFAULT_REGULARIZATIONS,
        n_estimators: int = 40,
        time_limit: int = 60,
    ):
        self.depth_budget = int(max_depth) + 1
        self.regularizations = tuple(regularizations)
        self.n_estimators = n_estimators
        self.time_limit = time_limit
        self.n_timeouts = 0
        self.n_failures = 0

    def build(self, X, y, seed) -> Tuple[List, int]:
        import pandas as pd
        from gosdt import GOSDTClassifier, Status, ThresholdGuessBinarizer

        _patch_sklearn_compat()
        binarizer = ThresholdGuessBinarizer(
            n_estimators=self.n_estimators, max_depth=1, random_state=seed
        )
        binarizer.set_output(transform="pandas")
        X_bin = binarizer.fit_transform(pd.DataFrame(X), y)

        models = []
        for regularization in self.regularizations:
            model = GOSDTClassifier(
                regularization=regularization,
                depth_budget=self.depth_budget,
                time_limit=self.time_limit,
                allow_small_reg=True,
            )
            try:
                model.fit(X_bin, y)
            except RuntimeError:
                # GOSDT occasionally reports "false convergence, no model was
                # found" for one penalty. Dropping that point from the path is
                # the conservative choice: it can only lower GOSDT's hypervolume.
                self.n_failures += 1
                continue
            if model.result_.status == Status.TIMEOUT:
                self.n_timeouts += 1
            models.append(_BinarisedGOSDT(binarizer, model))
        # An exact solver: nothing is sampled and discarded, so, like CART's
        # pruning path, there is no search budget to match.
        return models, 0
