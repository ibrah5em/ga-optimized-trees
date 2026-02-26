"""Test GA operators."""

import numpy as np

from ga_trees.ga.engine import Crossover, GAConfig, GAEngine, Mutation, Selection, TreeInitializer
from ga_trees.genotype.tree_genotype import TreeGenotype, create_internal_node, create_leaf_node


class TestSelection:
    """Test selection operators."""

    def create_population(self, n=10):
        """Create dummy population."""
        pop = []
        for i in range(n):
            left = create_leaf_node(0, 1)
            right = create_leaf_node(1, 1)
            root = create_internal_node(0, 0.5, left, right, 0)
            tree = TreeGenotype(root=root, n_features=4, n_classes=2)
            tree.fitness_ = float(i) / n  # Increasing fitness
            pop.append(tree)
        return pop

    def test_tournament_selection(self):
        """Test tournament selection."""
        pop = self.create_population(10)
        selected = Selection.tournament_selection(pop, tournament_size=3, n_select=5)

        assert len(selected) == 5
        # Higher fitness should be more likely selected
        avg_fitness = np.mean([t.fitness_ for t in selected])
        assert avg_fitness > 0.3  # Should be above random average

    def test_elitism_selection(self):
        """Test elitism selection."""
        pop = self.create_population(10)
        elite = Selection.elitism_selection(pop, n_elite=3)

        assert len(elite) == 3
        # Should get top 3
        fitnesses = [t.fitness_ for t in elite]
        assert fitnesses == sorted(fitnesses, reverse=True)
        assert fitnesses[0] >= 0.9  # Best individual


class TestCrossover:
    """Test crossover operators."""

    def create_tree(self, depth=2):
        """Create a tree of given depth."""
        if depth == 0:
            return create_leaf_node(0, 0)
        left = self.create_tree(depth - 1)
        right = self.create_tree(depth - 1)
        return create_internal_node(0, 0.5, left, right, 0)

    def test_subtree_crossover(self):
        """Test subtree crossover."""
        root1 = self.create_tree(2)
        root2 = self.create_tree(2)

        tree1 = TreeGenotype(root=root1, n_features=4, n_classes=2, max_depth=5)
        tree2 = TreeGenotype(root=root2, n_features=4, n_classes=2, max_depth=5)

        child1, child2 = Crossover.subtree_crossover(tree1, tree2)

        assert child1 is not tree1
        assert child2 is not tree2
        assert child1.get_depth() <= child1.max_depth
        assert child2.get_depth() <= child2.max_depth


class TestMutation:
    """Test mutation operators."""

    def test_threshold_perturbation(self):
        """Test threshold mutation."""
        left = create_leaf_node(0, 1)
        right = create_leaf_node(1, 1)
        root = create_internal_node(0, 0.5, left, right, 0)
        tree = TreeGenotype(root=root, n_features=4, n_classes=2)

        feature_ranges = {i: (0.0, 1.0) for i in range(4)}
        mutation = Mutation(n_features=4, feature_ranges=feature_ranges)

        original_threshold = tree.root.threshold
        mutated = mutation.threshold_perturbation(tree)

        assert mutated.root.threshold != original_threshold
        assert 0.0 <= mutated.root.threshold <= 1.0

    def test_prune_subtree(self):
        """Test pruning mutation."""
        # Create tree with internal nodes
        l1 = create_leaf_node(0, 2)
        l2 = create_leaf_node(1, 2)
        internal = create_internal_node(1, 0.5, l1, l2, 1)
        leaf = create_leaf_node(0, 1)
        root = create_internal_node(0, 0.3, internal, leaf, 0)

        tree = TreeGenotype(root=root, n_features=4, n_classes=2, max_depth=5)
        original_nodes = tree.get_num_nodes()

        feature_ranges = {i: (0.0, 1.0) for i in range(4)}
        mutation = Mutation(n_features=4, feature_ranges=feature_ranges)

        mutated = mutation.prune_subtree(tree)

        # Tree should be smaller or same size
        assert mutated.get_num_nodes() <= original_nodes


class TestTreeInitializer:
    """Test tree initialization."""

    def test_create_random_tree(self):
        """Test random tree creation."""
        X = np.random.rand(100, 4)
        y = np.random.randint(0, 2, 100)

        initializer = TreeInitializer(
            n_features=4, n_classes=2, max_depth=5, min_samples_split=10, min_samples_leaf=5
        )

        tree = initializer.create_random_tree(X, y)

        assert tree.get_depth() <= 5
        assert tree.get_num_nodes() >= 1
        valid, errors = tree.validate()
        assert valid, f"Tree validation failed: {errors}"


def test_early_stopping(iris_dataset):
    """GA stops early when fitness stagnates."""
    X, y = iris_dataset
    n_features = X.shape[1]
    n_classes = len(np.unique(y))
    feature_ranges = {i: (float(X[:, i].min()), float(X[:, i].max())) for i in range(n_features)}

    from ga_trees.fitness.calculator import FitnessCalculator

    config = GAConfig(
        population_size=20,
        n_generations=200,  # Very high — should stop early
        early_stopping_rounds=5,
        random_state=42,
    )
    initializer = TreeInitializer(
        n_features, n_classes, max_depth=3, min_samples_split=5, min_samples_leaf=2
    )
    fitness_calc = FitnessCalculator()
    mutation = Mutation(n_features, feature_ranges)
    engine = GAEngine(config, initializer, fitness_calc.calculate_fitness, mutation)

    engine.evolve(X, y, verbose=False)

    # Should have stopped well before 200 generations
    assert len(engine.history["best_fitness"]) < 200


# ---------------------------------------------------------------------------
# §3.2: New mutation operators
# ---------------------------------------------------------------------------


def _depth2_tree(n_features=4):
    """Create a depth-2 tree for mutation tests."""
    ll = create_leaf_node(0, 2)
    lr = create_leaf_node(1, 2)
    rl = create_leaf_node(0, 2)
    rr = create_leaf_node(1, 2)
    left = create_internal_node(1, 0.3, ll, lr, depth=1)
    right = create_internal_node(2, 0.7, rl, rr, depth=1)
    root = create_internal_node(0, 0.5, left, right, depth=0)
    return TreeGenotype(root=root, n_features=n_features, n_classes=2, max_depth=5)


class TestNewMutationOperators:
    def _make_mutation(self):
        return Mutation(n_features=4, feature_ranges={i: (0.0, 1.0) for i in range(4)})

    # subtree_regeneration ---------------------------------------------------

    def test_subtree_regeneration_returns_valid_tree(self):
        tree = _depth2_tree()
        mutation = self._make_mutation()
        result = mutation.subtree_regeneration(tree)
        assert isinstance(result, TreeGenotype)

    def test_subtree_regeneration_on_leaf_only_tree(self):
        """subtree_regeneration on a root-only internal tree with no non-root candidates."""
        left = create_leaf_node(0, 1)
        right = create_leaf_node(1, 1)
        root = create_internal_node(0, 0.5, left, right, depth=0)
        tree = TreeGenotype(root=root, n_features=4, n_classes=2, max_depth=5)
        mutation = self._make_mutation()
        # Candidates exclude root → candidates = [] → returns unchanged
        result = mutation.subtree_regeneration(tree)
        assert result is not None

    def test_subtree_regeneration_no_feature_ranges(self):
        """subtree_regeneration with empty feature_ranges uses threshold=0.0 branch."""
        tree = _depth2_tree()
        mutation = Mutation(n_features=4, feature_ranges={})
        result = mutation.subtree_regeneration(tree)
        assert result is not None

    # swap_children ----------------------------------------------------------

    def test_swap_children_changes_subtrees(self):
        tree = _depth2_tree()
        mutation = self._make_mutation()
        original_left_id = tree.root.left_child.node_id
        result = mutation.swap_children(tree)
        # Root's new left child was the old right child
        assert result.root.left_child.node_id != original_left_id

    def test_swap_children_on_leaf_tree(self):
        root = create_leaf_node(0, depth=0)
        tree = TreeGenotype(root=root, n_features=4, n_classes=2, max_depth=5)
        mutation = self._make_mutation()
        result = mutation.swap_children(tree)
        assert result.root.is_leaf()

    # hoist_mutation ---------------------------------------------------------

    def test_hoist_mutation_reduces_depth(self):
        tree = _depth2_tree()
        mutation = self._make_mutation()
        original_depth = tree.get_depth()
        result = mutation.hoist_mutation(tree)
        # After hoisting a subtree, depth should be <= original
        assert result.get_depth() <= original_depth

    def test_hoist_mutation_on_leaf_tree_unchanged(self):
        root = create_leaf_node(0, depth=0)
        tree = TreeGenotype(root=root, n_features=4, n_classes=2, max_depth=5)
        mutation = self._make_mutation()
        result = mutation.hoist_mutation(tree)
        assert result.root.is_leaf()

    def test_hoist_mutation_produces_valid_tree(self):
        tree = _depth2_tree()
        mutation = self._make_mutation()
        result = mutation.hoist_mutation(tree)
        valid, errs = result.validate()
        assert valid, f"hoist_mutation produced invalid tree: {errs}"

    # smart_threshold --------------------------------------------------------

    def test_smart_threshold_stays_in_range(self):
        left = create_leaf_node(0, 1)
        right = create_leaf_node(1, 1)
        root = create_internal_node(0, 0.5, left, right, depth=0)
        tree = TreeGenotype(root=root, n_features=4, n_classes=2, max_depth=5)
        mutation = self._make_mutation()
        for _ in range(10):
            result = mutation.smart_threshold(tree.copy())
            assert 0.0 <= result.root.threshold <= 1.0

    def test_smart_threshold_on_leaf_tree_unchanged(self):
        root = create_leaf_node(0, depth=0)
        tree = TreeGenotype(root=root, n_features=4, n_classes=2, max_depth=5)
        mutation = self._make_mutation()
        result = mutation.smart_threshold(tree)
        assert result.root.is_leaf()

    # _generate_random_subtree -----------------------------------------------

    def test_generate_random_subtree_returns_node(self):
        mutation = self._make_mutation()
        node = mutation._generate_random_subtree(0, 3)
        assert node is not None

    def test_generate_random_subtree_at_max_depth_returns_leaf(self):
        mutation = self._make_mutation()
        node = mutation._generate_random_subtree(3, 3)
        assert node.is_leaf()

    # mutate dispatch for new types ------------------------------------------

    def test_mutate_dispatches_swap_children(self):
        tree = _depth2_tree()
        mutation = self._make_mutation()
        result = mutation.mutate(tree, {"swap_children": 1.0})
        assert result is not None

    def test_mutate_dispatches_hoist(self):
        tree = _depth2_tree()
        mutation = self._make_mutation()
        result = mutation.mutate(tree, {"hoist": 1.0})
        assert result is not None

    def test_mutate_dispatches_subtree_regeneration(self):
        tree = _depth2_tree()
        mutation = self._make_mutation()
        result = mutation.mutate(tree, {"subtree_regeneration": 1.0})
        assert result is not None

    def test_mutate_dispatches_smart_threshold(self):
        left = create_leaf_node(0, 1)
        right = create_leaf_node(1, 1)
        root = create_internal_node(0, 0.5, left, right, depth=0)
        tree = TreeGenotype(root=root, n_features=4, n_classes=2, max_depth=5)
        mutation = self._make_mutation()
        result = mutation.mutate(tree, {"smart_threshold": 1.0})
        assert result is not None
