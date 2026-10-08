"""Unit tests for the nested CV benchmark harness.

Covers:
- FittedModel / complexity reporting
- sklearn_tree_complexity, ga_tree_complexity
- ga_evaluation_budget (the GA vs random search budget-matching formula)
- select_hyperparameters (inner CV, no outer leakage)
- run_nested_cv (fold indexing, seeding, pairing)
- verify_budget_match
- results_to_nested_dict
"""

import numpy as np
import pytest
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier

from ga_trees.benchmark import (
    FoldResult,
    GATreeMethod,
    PrunedCARTMethod,
    RandomTreeSearch,
    UnconstrainedCARTMethod,
    results_to_nested_dict,
    run_nested_cv,
    select_hyperparameters,
    verify_budget_match,
)
from ga_trees.benchmark.methods import ga_evaluation_budget
from ga_trees.benchmark.protocol import (
    BenchmarkMethod,
    FittedModel,
    ga_tree_complexity,
    sklearn_tree_complexity,
)
from ga_trees.genotype.tree_genotype import TreeGenotype, create_internal_node, create_leaf_node
from ga_trees.reproducibility import derive_fold_seed

TREE_CONFIG = {
    "max_depth": 4,
    "min_samples_split": 5,
    "min_samples_leaf": 2,
    "growth_stop_prob": 0.3,
}
GA_CONFIG = {
    "population_size": 8,
    "n_generations": 2,
    "crossover_prob": 0.7,
    "mutation_prob": 0.2,
    "tournament_size": 3,
    "elitism_ratio": 0.25,
    "mutation_types": None,
}
FITNESS_CONFIG = {"weights": {"accuracy": 0.7, "interpretability": 0.3}}


@pytest.fixture
def iris_small():
    X, y = load_iris(return_X_y=True)
    return X, y


class _ConstantMethod(BenchmarkMethod):
    """Method with a tunable knob whose best value is known in advance."""

    name = "Constant"

    def __init__(self, best=2):
        self.best = best
        self.fitted_with = []

    def param_grid(self, X, y):
        return [{"k": 1}, {"k": 2}, {"k": 3}]

    def fit(self, X, y, params, seed):
        self.fitted_with.append((params.get("k"), len(X)))
        # Accuracy peaks at self.best so inner CV has a clear winner.
        correct = 1.0 if params.get("k") == self.best else 0.0
        majority = int(np.bincount(y.astype(int)).argmax())

        def predict(X_new):
            if correct:
                return np.full(len(X_new), majority)
            return np.full(len(X_new), -1)

        return FittedModel(predict, 1, 1, 0, 0, 0.0)


# ---------------------------------------------------------------------------
# FittedModel
# ---------------------------------------------------------------------------


class TestFittedModel:
    def test_complexity_keys(self):
        model = FittedModel(lambda X: X, 7, 4, 3, 2, 1.5, n_evaluations=10)
        assert model.complexity() == {
            "nodes": 7,
            "leaves": 4,
            "depth": 3,
            "features_used": 2,
            "mean_path_length": 1.5,
            "evaluations": 10,
        }

    def test_predict_delegates(self):
        model = FittedModel(lambda X: np.zeros(len(X)), 1, 1, 0, 0, 0.0)
        assert list(model.predict(np.ones((3, 2)))) == [0.0, 0.0, 0.0]

    def test_evaluations_default_zero(self):
        assert FittedModel(lambda X: X, 1, 1, 0, 0, 0.0).n_evaluations == 0


# ---------------------------------------------------------------------------
# Complexity measures
# ---------------------------------------------------------------------------


class TestComplexityMeasures:
    def test_sklearn_stump(self, iris_small):
        X, y = iris_small
        estimator = DecisionTreeClassifier(max_depth=1, random_state=0).fit(X, y)
        measures = sklearn_tree_complexity(estimator, X)
        assert measures["n_nodes"] == 3
        assert measures["n_leaves"] == 2
        assert measures["max_depth"] == 1
        assert measures["n_features_used"] == 1
        assert measures["mean_path_length"] == pytest.approx(1.0)

    def test_sklearn_deeper_tree_has_longer_paths(self, iris_small):
        X, y = iris_small
        shallow = DecisionTreeClassifier(max_depth=1, random_state=0).fit(X, y)
        deep = DecisionTreeClassifier(max_depth=None, random_state=0).fit(X, y)
        assert (
            sklearn_tree_complexity(deep, X)["mean_path_length"]
            > sklearn_tree_complexity(shallow, X)["mean_path_length"]
        )

    def test_ga_leaf_only_tree(self):
        tree = TreeGenotype(root=create_leaf_node(0, 0), n_features=4, n_classes=2)
        measures = ga_tree_complexity(tree, np.random.rand(10, 4))
        assert measures["n_nodes"] == 1
        assert measures["n_leaves"] == 1
        assert measures["mean_path_length"] == pytest.approx(0.0)

    def test_ga_stump_path_length_is_one(self):
        root = create_internal_node(0, 0.5, create_leaf_node(0, 1), create_leaf_node(1, 1), 0)
        tree = TreeGenotype(root=root, n_features=4, n_classes=2)
        measures = ga_tree_complexity(tree, np.random.rand(20, 4))
        assert measures["n_leaves"] == 2
        assert measures["mean_path_length"] == pytest.approx(1.0)

    def test_ga_and_sklearn_agree_on_a_stump(self, iris_small):
        """Both measure path length the same way, or the comparison is meaningless."""
        X, y = iris_small
        estimator = DecisionTreeClassifier(max_depth=1, random_state=0).fit(X, y)
        root = create_internal_node(2, 2.45, create_leaf_node(0, 1), create_leaf_node(1, 1), 0)
        ga_tree = TreeGenotype(root=root, n_features=4, n_classes=3)
        assert ga_tree_complexity(ga_tree, X)["mean_path_length"] == pytest.approx(
            sklearn_tree_complexity(estimator, X)["mean_path_length"]
        )


# ---------------------------------------------------------------------------
# Budget matching
# ---------------------------------------------------------------------------


class TestEvaluationBudget:
    def test_accounts_for_elitism(self):
        """Elites keep their fitness, so a generation costs less than a full population."""
        config = {"population_size": 20, "n_generations": 5, "elitism_ratio": 0.1}
        # 20 initial + 5 generations x (20 - 2 elites)
        assert ga_evaluation_budget(config) == 110

    def test_is_not_the_naive_product(self):
        config = {"population_size": 20, "n_generations": 5, "elitism_ratio": 0.1}
        assert ga_evaluation_budget(config) != 20 * 5

    def test_zero_elitism_matches_full_replacement(self):
        config = {"population_size": 10, "n_generations": 3, "elitism_ratio": 0.0}
        assert ga_evaluation_budget(config) == 10 + 3 * 10

    def test_missing_elitism_defaults_to_zero(self):
        assert ga_evaluation_budget({"population_size": 10, "n_generations": 2}) == 30

    def test_ga_and_random_search_report_the_same_budget(self):
        ga = GATreeMethod(GA_CONFIG, TREE_CONFIG, FITNESS_CONFIG)
        rs = RandomTreeSearch(GA_CONFIG, TREE_CONFIG, FITNESS_CONFIG)
        assert ga.evaluation_budget({}) == rs.evaluation_budget({})

    def test_ga_spends_its_stated_budget(self, iris_small):
        """The formula must match what the engine actually does."""
        X, y = iris_small
        method = GATreeMethod(GA_CONFIG, TREE_CONFIG, FITNESS_CONFIG, tune=False)
        model = method.fit(X, y, {}, seed=1)
        assert model.n_evaluations == method.evaluation_budget({})

    def test_random_search_spends_the_same(self, iris_small):
        X, y = iris_small
        method = RandomTreeSearch(GA_CONFIG, TREE_CONFIG, FITNESS_CONFIG, tune=False)
        model = method.fit(X, y, {}, seed=1)
        assert model.n_evaluations == method.evaluation_budget({})


class TestVerifyBudgetMatch:
    def _result(self, method, evaluations):
        return FoldResult(
            dataset="d",
            method=method,
            fold=1,
            seed=0,
            test_accuracy=0.9,
            test_f1=0.9,
            fit_seconds=0.1,
            complexity={"evaluations": evaluations},
        )

    def test_equal_budgets_match(self):
        results = [self._result("A", 100), self._result("B", 100)]
        assert verify_budget_match(results, ["A", "B"])["matched"] is True

    def test_unequal_budgets_do_not_match(self):
        results = [self._result("A", 100), self._result("B", 50)]
        report = verify_budget_match(results, ["A", "B"])
        assert report["matched"] is False
        assert report["spread"] == pytest.approx(0.5)

    def test_within_tolerance_matches(self):
        results = [self._result("A", 100), self._result("B", 98)]
        assert verify_budget_match(results, ["A", "B"], tolerance=0.05)["matched"] is True

    def test_single_method_cannot_match(self):
        assert verify_budget_match([self._result("A", 100)], ["A"])["matched"] is False

    def test_reports_mean_evaluations(self):
        results = [self._result("A", 100), self._result("A", 200), self._result("B", 150)]
        means = verify_budget_match(results, ["A", "B"])["mean_evaluations"]
        assert means["A"] == pytest.approx(150.0)


# ---------------------------------------------------------------------------
# Inner-CV hyperparameter selection
# ---------------------------------------------------------------------------


class TestSelectHyperparameters:
    def test_picks_the_best_grid_point(self, iris_small):
        X, y = iris_small
        method = _ConstantMethod(best=2)
        assert select_hyperparameters(method, X, y, seed=0, inner_splits=3) == {"k": 2}

    def test_single_point_grid_skips_cv(self, iris_small):
        X, y = iris_small
        method = UnconstrainedCARTMethod()
        assert select_hyperparameters(method, X, y, seed=0, inner_splits=3) == {}

    def test_never_fits_on_the_full_training_split(self, iris_small):
        """Inner fits must see strictly less data than they were given."""
        X, y = iris_small
        method = _ConstantMethod()
        select_hyperparameters(method, X, y, seed=0, inner_splits=3)
        assert all(size < len(X) for _, size in method.fitted_with)

    def test_grid_is_fully_explored(self, iris_small):
        X, y = iris_small
        method = _ConstantMethod()
        select_hyperparameters(method, X, y, seed=0, inner_splits=3)
        assert {k for k, _ in method.fitted_with} == {1, 2, 3}

    def test_tiny_class_falls_back_to_first_grid_point(self):
        """A class with one member cannot be stratified; do not crash."""
        X = np.random.RandomState(0).rand(20, 3)
        y = np.array([0] * 19 + [1])
        method = _ConstantMethod()
        assert select_hyperparameters(method, X, y, seed=0, inner_splits=5) == {"k": 1}

    def test_cart_grid_is_data_dependent(self, iris_small):
        X, y = iris_small
        method = PrunedCARTMethod(TREE_CONFIG)
        grid = method.param_grid(X, y)
        assert len(grid) > 1
        assert {"ccp_alpha", "max_depth"} == set(grid[0])


# ---------------------------------------------------------------------------
# run_nested_cv
# ---------------------------------------------------------------------------


class TestRunNestedCV:
    def _methods(self):
        return [UnconstrainedCARTMethod(), PrunedCARTMethod(TREE_CONFIG)]

    def test_row_count(self, iris_small):
        X, y = iris_small
        results = run_nested_cv(
            X, y, self._methods(), "iris", outer_splits=3, outer_repeats=2, inner_splits=3
        )
        assert len(results) == 3 * 2 * 2

    def test_folds_are_one_indexed(self, iris_small):
        """Must match build_seed_manifest, which enumerates 1..n_folds."""
        X, y = iris_small
        results = run_nested_cv(
            X, y, [UnconstrainedCARTMethod()], "iris", outer_splits=3, outer_repeats=1
        )
        assert sorted(r.fold for r in results) == [1, 2, 3]

    def test_seeds_match_the_manifest_derivation(self, iris_small):
        X, y = iris_small
        results = run_nested_cv(
            X, y, [UnconstrainedCARTMethod()], "iris", base_seed=42, outer_splits=3, outer_repeats=1
        )
        for result in results:
            assert result.seed == derive_fold_seed(42, "iris", result.fold, result.method)

    def test_methods_get_different_seeds_on_the_same_fold(self, iris_small):
        """Sharing a random stream would correlate the methods' results."""
        X, y = iris_small
        results = run_nested_cv(
            X, y, self._methods(), "iris", outer_splits=3, outer_repeats=1, inner_splits=3
        )
        by_fold = {}
        for result in results:
            by_fold.setdefault(result.fold, []).append(result.seed)
        assert all(len(set(seeds)) == len(seeds) for seeds in by_fold.values())

    def test_all_methods_see_every_fold(self, iris_small):
        """Comparisons are paired, so no method may skip a fold."""
        X, y = iris_small
        results = run_nested_cv(
            X, y, self._methods(), "iris", outer_splits=3, outer_repeats=1, inner_splits=3
        )
        folds_per_method = {}
        for result in results:
            folds_per_method.setdefault(result.method, set()).add(result.fold)
        assert len(set(map(frozenset, folds_per_method.values()))) == 1

    def test_complexity_is_recorded(self, iris_small):
        X, y = iris_small
        results = run_nested_cv(
            X, y, [UnconstrainedCARTMethod()], "iris", outer_splits=3, outer_repeats=1
        )
        assert all(r.complexity["leaves"] > 0 for r in results)

    def test_reduces_folds_when_a_class_is_too_small(self):
        X = np.random.RandomState(0).rand(40, 3)
        y = np.array([0] * 36 + [1] * 4)
        results = run_nested_cv(
            X, y, [UnconstrainedCARTMethod()], "tiny", outer_splits=10, outer_repeats=1
        )
        assert len(results) == 4  # reduced to the smallest class size

    def test_progress_callback_is_called(self, iris_small):
        X, y = iris_small
        lines = []
        run_nested_cv(
            X,
            y,
            [UnconstrainedCARTMethod()],
            "iris",
            outer_splits=3,
            outer_repeats=1,
            progress=lines.append,
        )
        assert len(lines) == 3

    def test_no_methods_raises(self, iris_small):
        X, y = iris_small
        with pytest.raises(ValueError, match="at least one method"):
            run_nested_cv(X, y, [], "iris")

    def test_as_row_flattens_params_and_complexity(self, iris_small):
        X, y = iris_small
        results = run_nested_cv(
            X,
            y,
            [PrunedCARTMethod(TREE_CONFIG)],
            "iris",
            outer_splits=3,
            outer_repeats=1,
            inner_splits=3,
        )
        row = results[0].as_row()
        assert "ccp_alpha=" in row["selected_params"]
        assert "leaves" in row and "dataset" in row


class TestResultsToNestedDict:
    def test_shape_matches_statistics_layer(self, iris_small):
        X, y = iris_small
        results = run_nested_cv(
            X, y, [UnconstrainedCARTMethod()], "iris", outer_splits=3, outer_repeats=1
        )
        nested = results_to_nested_dict(results)
        assert set(nested) == {"iris"}
        assert "test_acc" in nested["iris"]["CART (unconstrained)"]
        assert len(nested["iris"]["CART (unconstrained)"]["test_acc"]) == 3

    def test_consumable_by_per_dataset_means(self, iris_small):
        from ga_trees.evaluation.statistics import per_dataset_means

        X, y = iris_small
        results = run_nested_cv(
            X, y, [UnconstrainedCARTMethod()], "iris", outer_splits=3, outer_repeats=1
        )
        datasets, scores = per_dataset_means(results_to_nested_dict(results))
        assert datasets == ["iris"]
        assert len(scores["CART (unconstrained)"]) == 1


class TestSearchGrid:
    """The GA and random search must always be tuned over the same grid."""

    def test_weight_only_grid_leaves_depth_to_the_config(self):
        from ga_trees.benchmark.methods import search_grid

        grid = search_grid(tune_depth=False)
        assert [p["accuracy_weight"] for p in grid] == [0.5, 0.7, 0.9]
        assert all("max_depth" not in p for p in grid)
        assert len(search_grid(tune_depth=True)) == 9

    def test_ga_and_random_search_share_the_grid(self):
        import numpy as np

        from ga_trees.benchmark.methods import GATreeMethod, RandomTreeSearch

        cfg = {"population_size": 4, "n_generations": 2}
        tree = {"max_depth": 6, "min_samples_split": 8, "min_samples_leaf": 3}
        X, y = np.zeros((10, 2)), np.zeros(10)
        for tune_depth in (True, False):
            ga = GATreeMethod(cfg, tree, {}, tune_depth=tune_depth)
            rs = RandomTreeSearch(cfg, tree, {}, tune_depth=tune_depth)
            assert ga.param_grid(X, y) == rs.param_grid(X, y)
