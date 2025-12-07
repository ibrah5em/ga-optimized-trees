# Custom Genetic Operators

Guide to creating custom mutation, crossover, and selection operators.

## Overview

Genetic operators drive the evolution process. Custom operators allow you to:
- Implement domain-specific mutations
- Create specialized crossover strategies
- Design custom selection mechanisms
- Incorporate problem-specific knowledge

## Custom Mutation Operators

### Basic Structure

```python
from ga_trees.ga.engine import Mutation
from ga_trees.genotype.tree_genotype import create_leaf_node, create_internal_node

class CustomMutation(Mutation):
    def __init__(self, n_features, feature_ranges):
        super().__init__(n_features, feature_ranges)
    
    def mutate(self, tree, mutation_types):
        """Override to add custom mutations."""
        # Add your custom mutation type
        extended_types = {**mutation_types, 'custom_mutation': 0.15}
        
        # Select mutation
        import random
        mut_type = random.choices(
            list(extended_types.keys()),
            weights=list(extended_types.values()),
            k=1
        )[0]
        
        if mut_type == 'custom_mutation':
            return self.custom_mutation(tree)
        else:
            return super().mutate(tree, mutation_types)
    
    def custom_mutation(self, tree):
        """Your custom mutation logic."""
        # Example: swap two random internal nodes
        tree = tree.copy()
        internal_nodes = tree.get_internal_nodes()
        
        if len(internal_nodes) >= 2:
            node1, node2 = random.sample(internal_nodes, 2)
            # Swap feature indices
            node1.feature_idx, node2.feature_idx = node2.feature_idx, node1.feature_idx
            node1.threshold, node2.threshold = node2.threshold, node1.threshold
        
        return tree
```

### Example: Feature-Aware Mutation

```python
class FeatureGroupMutation(Mutation):
    """Mutation that respects feature groups."""
    
    def __init__(self, n_features, feature_ranges, feature_groups):
        """
        Args:
            feature_groups: Dict mapping group_id to list of feature indices
                Example: {0: [0, 1, 2], 1: [3, 4], 2: [5, 6, 7]}
        """
        super().__init__(n_features, feature_ranges)
        self.feature_groups = feature_groups
    
    def group_aware_replacement(self, tree):
        """Replace feature with one from same group."""
        tree = tree.copy()
        internal_nodes = tree.get_internal_nodes()
        
        if not internal_nodes:
            return tree
        
        import random
        node = random.choice(internal_nodes)
        current_feature = node.feature_idx
        
        # Find group
        current_group = None
        for group_id, features in self.feature_groups.items():
            if current_feature in features:
                current_group = group_id
                break
        
        if current_group is not None:
            # Replace with feature from same group
            group_features = self.feature_groups[current_group]
            new_feature = random.choice([f for f in group_features if f != current_feature])
            node.feature_idx = new_feature
            
            # Update threshold
            if new_feature in self.feature_ranges:
                min_val, max_val = self.feature_ranges[new_feature]
                node.threshold = random.uniform(min_val, max_val)
        
        return tree
```

## Custom Crossover Operators

### Basic Structure

```python
from ga_trees.ga.engine import Crossover

class CustomCrossover(Crossover):
    @staticmethod
    def balanced_crossover(parent1, parent2):
        """Crossover that maintains tree balance."""
        child1 = parent1.copy()
        child2 = parent2.copy()
        
        # Select crossover points at similar depths
        nodes1 = child1.get_all_nodes()
        nodes2 = child2.get_all_nodes()
        
        # Group by depth
        from collections import defaultdict
        depth_groups1 = defaultdict(list)
        depth_groups2 = defaultdict(list)
        
        for node in nodes1[1:]:  # Skip root
            depth_groups1[node.depth].append(node)
        for node in nodes2[1:]:
            depth_groups2[node.depth].append(node)
        
        # Find common depths
        common_depths = set(depth_groups1.keys()) & set(depth_groups2.keys())
        
        if common_depths:
            import random
            depth = random.choice(list(common_depths))
            node1 = random.choice(depth_groups1[depth])
            node2 = random.choice(depth_groups2[depth])
            
            # Swap subtrees
            Crossover._copy_node_contents(node2, node1)
            Crossover._copy_node_contents(node1, node2)
        
        # Repair
        child1 = Crossover._repair_tree(child1)
        child2 = Crossover._repair_tree(child2)
        
        return child1, child2
```

## Custom Selection Operators

```python
class CustomSelection:
    @staticmethod
    def fitness_sharing_selection(population, tournament_size, n_select, niche_radius=0.1):
        """Selection with fitness sharing to maintain diversity."""
        import random
        import numpy as np
        
        def distance(tree1, tree2):
            """Distance metric between trees."""
            # Simple: difference in structure
            return abs(tree1.get_num_nodes() - tree2.get_num_nodes())
        
        # Calculate shared fitness
        shared_fitness = []
        for i, ind in enumerate(population):
            sharing_sum = sum(
                1 - min(distance(ind, other) / niche_radius, 1.0)
                for other in population
            )
            shared = ind.fitness_ / max(sharing_sum, 1.0)
            shared_fitness.append(shared)
        
        # Tournament selection with shared fitness
        selected = []
        for _ in range(n_select):
            tournament_idx = random.sample(range(len(population)), tournament_size)
            winner_idx = max(tournament_idx, key=lambda i: shared_fitness[i])
            selected.append(population[winner_idx].copy())
        
        return selected
```

## Integrating Custom Operators

```python
from ga_trees.ga.engine import GAEngine, GAConfig, TreeInitializer

# Create custom operators
custom_mutation = CustomMutation(n_features=n_features, feature_ranges=feature_ranges)
custom_selection = CustomSelection()

# Modify GA Engine to use custom operators
class CustomGAEngine(GAEngine):
    def __init__(self, config, initializer, fitness_function, mutation, selection=None):
        super().__init__(config, initializer, fitness_function, mutation)
        self.custom_selection = selection or Selection
    
    def evolve(self, X, y, verbose=True):
        """Modified evolution with custom operators."""
        self.initialize_population(X, y)
        self.evaluate_population(X, y)
        
        for generation in range(self.config.n_generations):
            # Use custom selection
            if self.custom_selection:
                parents = self.custom_selection.fitness_sharing_selection(
                    self.population,
                    self.config.tournament_size,
                    n_select=2
                )
            else:
                parents = Selection.tournament_selection(
                    self.population,
                    self.config.tournament_size,
                    n_select=2
                )
            
            # Rest of evolution...
            # (similar to standard GA)
        
        return self.best_individual

# Use custom engine
ga_engine = CustomGAEngine(
    config=ga_config,
    initializer=initializer,
    fitness_function=fitness_calc.calculate_fitness,
    mutation=custom_mutation,
    selection=custom_selection
)
```