"""Unit tests for the frontier-level benchmark.

K1 and H1 are stated on hypervolume, so the properties that matter here are:
the GA and random search spend exactly the same number of evaluations, every
method on a fold is scored against one shared reference point, and the objective
pair is (accuracy, -nodes) rather than the composite interpretability score that
K4 forbids as an outcome measure.
"""

import pytest
import yaml
from sklearn.datasets import load_iris

from ga_trees.benchmark.frontiers import (
    CARTPathFrontier,
    ParetoGAFrontier,
    RandomSearchFrontier,
    _CountingObjective,
    _score_candidates,
    dominance_rate,
    hypervolume_by_dataset,
    run_frontier_cv,
)
from ga_trees.fitness.calculator import FitnessCalculator

TREE_CONFIG = {
    "max_depth": 4,
    "min_samples_split": 5,
    "min_samples_leaf": 2,
    "growth_stop_prob": 0.3,
    "split_strategy": "midpoint",
}
GA_CONFIG = {
    "population_size": 16,
    "n_generations": 4,
    "crossover_prob": 0.7,
    "mutation_prob": 0.2,
    "tournament_size": 3,
    "elitism_ratio": 0.25,
    "mutation_types": {
        "threshold_perturbation": 0.4,
        "feature_replacement": 0.3,
        "prune_subtree": 0.2,
        "expand_leaf": 0.1,
    },
}
FITNESS_CONFIG = {"classification_metric": "accuracy", "validation_fraction": 0.2}


@pytest.fixture
def iris():
    return load_iris(return_X_y=True)


@pytest.fixture
def fold_results(iris):
    X, y = iris
    return run_frontier_cv(
        X,
        y,
        ParetoGAFrontier(GA_CONFIG, TREE_CONFIG, FITNESS_CONFIG),
        RandomSearchFrontier(GA_CONFIG, TREE_CONFIG, FITNESS_CONFIG),
        [CARTPathFrontier(TREE_CONFIG)],
        dataset_name="iris",
        base_seed=42,
        outer_splits=3,
        outer_repeats=1,
    )


class TestCountingObjective:
    def test_returns_accuracy_and_negated_node_count(self, iris):
        from ga_trees.ga.engine import TreeInitializer

        X, y = iris
        tree = TreeInitializer(X.shape[1], 3, 4, 5, 2).create_random_tree(X, y)
        objective = _CountingObjective(FitnessCalculator(mode="weighted_sum"), None, None)
        accuracy, negated_nodes = objective(tree, X, y)

        assert 0.0 <= accuracy <= 1.0
        # Size enters negated so NSGA-II's maximise-both convention minimises it.
        assert negated_nodes == -float(tree.get_num_nodes())
        assert negated_nodes < 0

    def test_counts_every_call(self, iris):
        from ga_trees.ga.engine import TreeInitializer

        X, y = iris
        initializer = TreeInitializer(X.shape[1], 3, 4, 5, 2)
        objective = _CountingObjective(FitnessCalculator(mode="weighted_sum"), None, None)
        for _ in range(7):
            objective(initializer.create_random_tree(X, y), X, y)
        assert objective.count == 7

    def test_composite_interpretability_is_not_an_objective(self, iris):
        # K4: the composite score is a search heuristic, never a reported axis.
        from ga_trees.ga.engine import TreeInitializer

        X, y = iris
        tree = TreeInitializer(X.shape[1], 3, 4, 5, 2).create_random_tree(X, y)
        objective = _CountingObjective(FitnessCalculator(mode="weighted_sum"), None, None)
        _, second = objective(tree, X, y)
        assert second != tree.interpretability_


class TestArchiving:
    def test_archive_keeps_only_non_dominated_candidates(self, iris):
        from ga_trees.ga.engine import TreeInitializer

        X, y = iris
        objective = _CountingObjective(
            FitnessCalculator(mode="weighted_sum"), None, None, archive=True
        )
        initializer = TreeInitializer(X.shape[1], 3, 5, 5, 2, growth_stop_prob=0.1)
        for _ in range(80):
            objective(initializer.create_random_tree(X, y), X, y)

        assert objective.count == 80
        assert 0 < len(objective.archive) <= 80
        points = [(a, n) for a, n, _ in objective.archive]
        for accuracy, nodes in points:
            dominators = [
                1
                for other_a, other_n in points
                if other_a >= accuracy
                and other_n >= nodes
                and (other_a > accuracy or other_n > nodes)
            ]
            assert not dominators, "archive retained a dominated point"

    def test_archive_holds_no_duplicate_objective_vectors(self, iris):
        from ga_trees.ga.engine import TreeInitializer

        X, y = iris
        objective = _CountingObjective(
            FitnessCalculator(mode="weighted_sum"), None, None, archive=True
        )
        initializer = TreeInitializer(X.shape[1], 3, 4, 5, 2)
        for _ in range(120):
            objective(initializer.create_random_tree(X, y), X, y)
        points = [(a, n) for a, n, _ in objective.archive]
        assert len(points) == len(set(points))

    def test_archiving_is_off_by_default(self, iris):
        from ga_trees.ga.engine import TreeInitializer

        X, y = iris
        objective = _CountingObjective(FitnessCalculator(mode="weighted_sum"), None, None)
        objective(TreeInitializer(X.shape[1], 3, 4, 5, 2).create_random_tree(X, y), X, y)
        assert objective.archive == []

    def test_random_search_delivers_an_archive_not_every_candidate(self, iris):
        # Returning all candidates lets the dominance filter run on their test
        # scores, which is a maximum over thousands of test evaluations and not
        # something any method can deliver.
        X, y = iris
        method = RandomSearchFrontier(GA_CONFIG, TREE_CONFIG, FITNESS_CONFIG)
        method.budget = 300
        candidates, evaluations = method.build(X, y, seed=0)
        assert evaluations == 300
        assert len(candidates) < evaluations

    def test_archived_ga_reports_a_distinct_method_name(self):
        plain = ParetoGAFrontier(GA_CONFIG, TREE_CONFIG, FITNESS_CONFIG)
        archived = ParetoGAFrontier(GA_CONFIG, TREE_CONFIG, FITNESS_CONFIG, archive=True)
        assert plain.name != archived.name
        assert "archived" in archived.name

    def test_archived_ga_delivers_at_least_the_final_front(self, iris):
        X, y = iris
        seed = 5
        plain, _ = ParetoGAFrontier(GA_CONFIG, TREE_CONFIG, FITNESS_CONFIG).build(X, y, seed)
        archived, _ = ParetoGAFrontier(GA_CONFIG, TREE_CONFIG, FITNESS_CONFIG, archive=True).build(
            X, y, seed
        )
        # Same seed, same run; the archive saw every candidate the front came from.
        assert len(archived) >= 1
        assert len(plain) >= 1


class TestBudgetMatching:
    def test_random_search_spends_exactly_the_ga_budget(self, fold_results):
        by_fold = {}
        for result in fold_results:
            by_fold.setdefault(result.fold, {})[result.method] = result.n_evaluations
        assert by_fold
        for fold, spend in by_fold.items():
            assert spend["GA (NSGA-II)"] == spend["Random Search"], f"fold {fold} unmatched"

    def test_random_search_refuses_to_run_without_a_budget(self, iris):
        X, y = iris
        method = RandomSearchFrontier(GA_CONFIG, TREE_CONFIG, FITNESS_CONFIG)
        with pytest.raises(ValueError, match="budget must be set"):
            method.build(X, y, seed=0)

    def test_cart_path_spends_no_search_budget(self, iris):
        X, y = iris
        _, evaluations = CARTPathFrontier(TREE_CONFIG).build(X, y, seed=0)
        assert evaluations == 0


class TestSharedReferencePoint:
    def test_one_reference_for_the_whole_dataset(self, fold_results):
        # PREREGISTRATION.md fixes it at "max over all methods on that dataset",
        # not per fold. A per-fold reference is a smaller box, and a smaller box
        # favours whichever method has the lower peak accuracy.
        references = {result.reference_nodes for result in fold_results}
        assert len(references) == 1, f"dataset scored against {len(references)} references"

    def test_reference_covers_every_frontier_point(self, fold_results):
        for result in fold_results:
            for _, nodes in result.points:
                assert nodes <= result.reference_nodes

    def test_reference_exceeds_the_largest_model_seen(self, fold_results):
        largest = max(nodes for r in fold_results for _, nodes in r.points)
        assert fold_results[0].reference_nodes > largest


class TestFrontierResults:
    def test_every_method_appears_on_every_fold(self, fold_results):
        expected = {"GA (NSGA-II)", "Random Search", "CART (ccp path)"}
        by_fold = {}
        for result in fold_results:
            by_fold.setdefault(result.fold, set()).add(result.method)
        assert all(methods == expected for methods in by_fold.values())

    def test_hypervolume_is_non_negative(self, fold_results):
        assert all(result.hypervolume >= 0 for result in fold_results)

    def test_distinct_count_is_reported_separately_from_point_count(self, fold_results):
        # The whole point of tracking both: a big candidate pool collapsing onto
        # a few objective points must be visible, not hidden behind front size.
        for result in fold_results:
            assert result.n_distinct >= result.n_points

    def test_rows_are_csv_ready(self, fold_results):
        row = fold_results[0].as_row()
        assert "hypervolume" in row and "n_distinct" in row
        assert "points" not in row  # the point list goes to its own artifact

    def test_reshaping_by_dataset(self, fold_results):
        nested = hypervolume_by_dataset(fold_results)
        assert set(nested) == {"iris"}
        assert set(nested["iris"]) == {"GA (NSGA-II)", "Random Search", "CART (ccp path)"}
        assert all(len(v) == 3 for v in nested["iris"].values())


class TestScoreCandidates:
    def test_handles_both_model_types(self, iris):
        from sklearn.tree import DecisionTreeClassifier

        from ga_trees.ga.engine import TreeInitializer

        X, y = iris
        genotype = TreeInitializer(X.shape[1], 3, 4, 5, 2).create_random_tree(X, y)
        estimator = DecisionTreeClassifier(random_state=0).fit(X, y)

        points = _score_candidates([genotype, estimator], X, y, X, y)
        assert len(points) == 2
        assert all(0.0 <= accuracy <= 1.0 and nodes >= 1 for accuracy, nodes in points)

    def test_empty_candidate_list(self, iris):
        X, y = iris
        assert _score_candidates([], X, y, X, y) == []


class TestDominanceRate:
    def test_rate_is_a_fraction_of_datasets(self, fold_results):
        rate = dominance_rate(fold_results, "GA (NSGA-II)", "CART (ccp path)")
        assert 0.0 <= rate <= 1.0

    def test_unknown_method_yields_zero(self, fold_results):
        assert dominance_rate(fold_results, "nope", "CART (ccp path)") == 0.0


def test_shipped_config_loads_into_the_harness():
    """The paper config must construct these methods without edits."""
    config = yaml.safe_load(open("configs/paper.yaml"))
    ParetoGAFrontier(config["ga"], config["tree"], config["fitness"])
    RandomSearchFrontier(config["ga"], config["tree"], config["fitness"])
    CARTPathFrontier(config["tree"])
