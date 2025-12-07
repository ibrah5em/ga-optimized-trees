# Custom Fitness Functions

Complete guide to creating and using custom fitness functions for domain-specific optimization.

## Overview

The fitness function is the core of the genetic algorithm - it determines what makes a "good" tree. Custom fitness functions allow you to:

- Optimize for domain-specific metrics (recall, precision, F2-score)
- Add domain constraints (max tree size, required features)
- Balance multiple objectives with custom weights
- Incorporate business logic into tree evolution

## Basic Fitness Structure

### Understanding the Default Fitness

```python
from ga_trees.fitness.calculator import FitnessCalculator

# Default weighted-sum fitness
fitness_calc = FitnessCalculator(
    mode='weighted_sum',
    accuracy_weight=0.68,
    interpretability_weight=0.32,
    interpretability_weights={
        'node_complexity': 0.50,
        'feature_coherence': 0.10,
        'tree_balance': 0.10,
        'semantic_coherence': 0.30
    }
)

# Fitness = 0.68 × accuracy + 0.32 × interpretability
```

## Creating Custom Fitness Functions

### Method 1: Extend FitnessCalculator

```python
from ga_trees.fitness.calculator import FitnessCalculator, TreePredictor
from sklearn.metrics import recall_score, precision_score
import numpy as np

class MedicalFitnessCalculator(FitnessCalculator):
    """
    Custom fitness for medical diagnosis:
    - Prioritize recall (sensitivity) over accuracy
    - Penalize false negatives heavily
    - Reward simple trees
    """
    
    def calculate_fitness(self, tree, X, y):
        # Fit leaf predictions
        self.predictor.fit_leaf_predictions(tree, X, y)
        y_pred = self.predictor.predict(tree, X)
        
        # Calculate recall (most important for medical)
        recall = recall_score(y, y_pred, average='weighted')
        
        # Calculate precision
        precision = precision_score(y, y_pred, average='weighted', zero_division=0)
        
        # F2-score (weights recall 2x more than precision)
        beta = 2.0
        f2 = (1 + beta**2) * (precision * recall) / (beta**2 * precision + recall + 1e-10)
        
        # Interpretability (medical requires simple explanations)
        interp = self.interp_calc.calculate_composite_score(
            tree, self.interpretability_weights
        )
        
        # Store individual scores
        tree.accuracy_ = f2
        tree.interpretability_ = interp
        
        # Weighted fitness: 70% F2-score + 30% interpretability
        fitness = 0.70 * f2 + 0.30 * interp
        
        return fitness
```

### Method 2: Standalone Function

```python
def credit_scoring_fitness(tree, X, y):
    """
    Custom fitness for credit scoring:
    - Minimize false positives (wrongly approve bad credit)
    - Require specific features (income, credit history)
    - Limit tree depth for regulatory compliance
    """
    from ga_trees.fitness.calculator import TreePredictor
    from sklearn.metrics import confusion_matrix
    
    predictor = TreePredictor()
    predictor.fit_leaf_predictions(tree, X, y)
    y_pred = predictor.predict(tree, X)
    
    # Get confusion matrix
    tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
    
    # False positive rate (critical for credit)
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    
    # True positive rate (also important)
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    # Overall accuracy
    accuracy = (tp + tn) / len(y)
    
    # Interpretability (regulatory requirement)
    node_penalty = 1.0 - min(tree.get_num_nodes() / 30.0, 1.0)
    depth_penalty = 1.0 - min(tree.get_depth() / 5.0, 1.0)
    
    # Feature requirement (must use income and credit_history)
    required_features = {0, 3}  # Feature indices
    features_used = tree.get_features_used()
    feature_bonus = 0.1 if required_features.issubset(features_used) else 0.0
    
    # Composite fitness
    fitness = (
        0.40 * (1.0 - fpr) +  # Minimize false positives
        0.30 * tpr +          # Maximize true positives
        0.15 * accuracy +     # Overall accuracy
        0.10 * node_penalty + # Small tree
        0.05 * depth_penalty + # Shallow tree
        feature_bonus         # Bonus for required features
    )
    
    # Store metrics
    tree.accuracy_ = accuracy
    tree.interpretability_ = (node_penalty + depth_penalty) / 2
    
    return fitness
```

### Method 3: Multi-Objective Custom Fitness

```python
class MultiObjectiveFitness:
    """
    Custom multi-objective fitness balancing:
    - Accuracy on majority class
    - Accuracy on minority class
    - Tree interpretability
    """
    
    def __init__(self, minority_class_weight=2.0):
        self.minority_class_weight = minority_class_weight
        self.predictor = TreePredictor()
    
    def calculate_fitness(self, tree, X, y):
        self.predictor.fit_leaf_predictions(tree, X, y)
        y_pred = self.predictor.predict(tree, X)
        
        # Separate classes
        classes = np.unique(y)
        class_counts = [np.sum(y == c) for c in classes]
        minority_class = classes[np.argmin(class_counts)]
        
        # Accuracy on each class
        accuracies = {}
        for c in classes:
            mask = y == c
            if np.sum(mask) > 0:
                acc = np.mean(y_pred[mask] == y[mask])
                accuracies[c] = acc
        
        # Weighted accuracy (emphasize minority class)
        minority_acc = accuracies.get(minority_class, 0)
        majority_acc = np.mean([accuracies[c] for c in classes if c != minority_class])
        
        balanced_acc = (
            self.minority_class_weight * minority_acc + 
            majority_acc
        ) / (self.minority_class_weight + 1)
        
        # Interpretability
        interp = 1.0 / (1.0 + tree.get_num_nodes() / 15.0)
        
        # Store metrics
        tree.accuracy_ = balanced_acc
        tree.interpretability_ = interp
        
        # Fitness
        fitness = 0.75 * balanced_acc + 0.25 * interp
        
        return fitness
```

## Domain-Specific Examples

### 1. Healthcare: Maximize Sensitivity

```python
class HealthcareFitness(FitnessCalculator):
    """Prioritize detecting disease (high recall)."""
    
    def calculate_fitness(self, tree, X, y):
        self.predictor.fit_leaf_predictions(tree, X, y)
        y_pred = self.predictor.predict(tree, X)
        
        # Sensitivity (true positive rate) - critical for disease detection
        from sklearn.metrics import recall_score
        sensitivity = recall_score(y, y_pred, pos_label=1)
        
        # Specificity (true negative rate)
        from sklearn.metrics import confusion_matrix
        tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        # Balance both but prioritize sensitivity
        clinical_metric = 0.70 * sensitivity + 0.30 * specificity
        
        # Interpretability (doctors need to explain decisions)
        interp = self.interp_calc.calculate_composite_score(
            tree, {
                'node_complexity': 0.70,  # Very important
                'feature_coherence': 0.20,
                'tree_balance': 0.05,
                'semantic_coherence': 0.05
            }
        )
        
        tree.accuracy_ = clinical_metric
        tree.interpretability_ = interp
        
        # Strong emphasis on interpretability for medical use
        return 0.60 * clinical_metric + 0.40 * interp
```

### 2. Finance: Minimize Risk

```python
class FinancialRiskFitness:
    """Minimize financial risk from misclassification."""
    
    def __init__(self, cost_fp=1000, cost_fn=100):
        """
        Args:
            cost_fp: Cost of false positive (approve bad loan)
            cost_fn: Cost of false negative (reject good loan)
        """
        self.cost_fp = cost_fp
        self.cost_fn = cost_fn
        self.predictor = TreePredictor()
    
    def __call__(self, tree, X, y):
        self.predictor.fit_leaf_predictions(tree, X, y)
        y_pred = self.predictor.predict(tree, X)
        
        # Calculate costs
        from sklearn.metrics import confusion_matrix
        tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
        
        total_cost = (fp * self.cost_fp) + (fn * self.cost_fn)
        max_possible_cost = len(y) * self.cost_fp  # All false positives
        
        # Normalize to [0, 1], higher is better
        cost_score = 1.0 - (total_cost / max_possible_cost)
        
        # Interpretability (regulatory compliance)
        interp = 1.0 - min(tree.get_num_nodes() / 20.0, 1.0)
        
        tree.accuracy_ = cost_score
        tree.interpretability_ = interp
        
        # Balance financial performance with interpretability
        return 0.75 * cost_score + 0.25 * interp
```

### 3. Legal: Maximize Explainability

```python
class LegalFitness:
    """Maximize explainability for legal decisions."""
    
    def __init__(self):
        self.predictor = TreePredictor()
    
    def __call__(self, tree, X, y):
        self.predictor.fit_leaf_predictions(tree, X, y)
        y_pred = self.predictor.predict(tree, X)
        
        # Basic accuracy
        from sklearn.metrics import accuracy_score
        accuracy = accuracy_score(y, y_pred)
        
        # Explainability metrics
        num_rules = len(tree.to_rules())
        avg_rule_length = np.mean([len(rule.split('AND')) for rule in tree.to_rules()])
        
        # Explainability score (prefer fewer, shorter rules)
        explainability = (
            0.50 * (1.0 - min(num_rules / 10.0, 1.0)) +  # Fewer rules
            0.30 * (1.0 - min(avg_rule_length / 5.0, 1.0)) +  # Shorter rules
            0.20 * tree.get_tree_balance()  # Balanced tree
        )
        
        tree.accuracy_ = accuracy
        tree.interpretability_ = explainability
        
        # Very high emphasis on explainability for legal use
        return 0.50 * accuracy + 0.50 * explainability
```

## Using Custom Fitness

### In Training Script

```python
from ga_trees.ga.engine import GAEngine, GAConfig, TreeInitializer, Mutation

# Create custom fitness
custom_fitness = MedicalFitnessCalculator(
    accuracy_weight=0.60,
    interpretability_weight=0.40
)

# Setup GA
ga_config = GAConfig(population_size=80, n_generations=40)
initializer = TreeInitializer(n_features=n_features, n_classes=n_classes,
                              max_depth=6, min_samples_split=8)
mutation = Mutation(n_features=n_features, feature_ranges=feature_ranges)

# Create GA engine with custom fitness
ga_engine = GAEngine(
    config=ga_config,
    initializer=initializer,
    fitness_function=custom_fitness.calculate_fitness,  # Your custom function
    mutation=mutation
)

# Train
best_tree = ga_engine.evolve(X_train, y_train, verbose=True)
```

### With Configuration

```python
# Create fitness from config
def create_fitness_from_config(config):
    fitness_type = config.get('fitness_type', 'default')
    
    if fitness_type == 'medical':
        return MedicalFitnessCalculator(**config['fitness_params'])
    elif fitness_type == 'financial':
        return FinancialRiskFitness(**config['fitness_params'])
    elif fitness_type == 'legal':
        return LegalFitness()
    else:
        return FitnessCalculator(**config['fitness_params'])

# Use in training
fitness = create_fitness_from_config(config)
ga_engine = GAEngine(config, initializer, fitness.calculate_fitness, mutation)
```

## Advanced Techniques

### 1. Adaptive Fitness

```python
class AdaptiveFitness:
    """Fitness that changes during evolution."""
    
    def __init__(self, initial_accuracy_weight=0.80):
        self.accuracy_weight = initial_accuracy_weight
        self.generation = 0
    
    def __call__(self, tree, X, y):
        # Gradually increase interpretability emphasis
        if self.generation > 20:
            self.accuracy_weight = max(0.60, self.accuracy_weight - 0.01)
        
        # Calculate metrics
        predictor = TreePredictor()
        predictor.fit_leaf_predictions(tree, X, y)
        y_pred = predictor.predict(tree, X)
        accuracy = accuracy_score(y, y_pred)
        interp = 1.0 / (1.0 + tree.get_num_nodes() / 15.0)
        
        # Dynamic weighting
        fitness = (
            self.accuracy_weight * accuracy + 
            (1 - self.accuracy_weight) * interp
        )
        
        return fitness
    
    def on_generation_end(self):
        """Call this after each generation."""
        self.generation += 1
```

### 2. Constraint-Based Fitness

```python
def constrained_fitness(tree, X, y, constraints):
    """
    Fitness with hard constraints.
    
    Args:
        constraints: Dict with:
            - max_nodes: Maximum allowed nodes
            - max_depth: Maximum allowed depth
            - required_features: Set of required feature indices
            - forbidden_features: Set of forbidden feature indices
    """
    # Check hard constraints
    if tree.get_num_nodes() > constraints.get('max_nodes', float('inf')):
        return 0.0  # Invalid solution
    
    if tree.get_depth() > constraints.get('max_depth', float('inf')):
        return 0.0  # Invalid solution
    
    features_used = tree.get_features_used()
    required = constraints.get('required_features', set())
    if required and not required.issubset(features_used):
        return 0.0  # Missing required features
    
    forbidden = constraints.get('forbidden_features', set())
    if forbidden and features_used.intersection(forbidden):
        return 0.0  # Uses forbidden features
    
    # Calculate fitness normally if constraints satisfied
    predictor = TreePredictor()
    predictor.fit_leaf_predictions(tree, X, y)
    y_pred = predictor.predict(tree, X)
    accuracy = accuracy_score(y, y_pred)
    
    return accuracy
```

## Testing Custom Fitness

```python
def test_custom_fitness():
    """Test custom fitness function."""
    from sklearn.datasets import load_breast_cancer
    
    X, y = load_breast_cancer(return_X_y=True)
    
    # Create simple tree for testing
    from ga_trees.genotype.tree_genotype import create_leaf_node, create_internal_node, TreeGenotype
    left = create_leaf_node(0, 1)
    right = create_leaf_node(1, 1)
    root = create_internal_node(0, 0.5, left, right, 0)
    tree = TreeGenotype(root=root, n_features=X.shape[1], n_classes=2)
    
    # Test fitness
    fitness_calc = MedicalFitnessCalculator()
    fitness = fitness_calc.calculate_fitness(tree, X, y)
    
    print(f"Fitness: {fitness:.4f}")
    print(f"Accuracy: {tree.accuracy_:.4f}")
    print(f"Interpretability: {tree.interpretability_:.4f}")
    
    assert 0.0 <= fitness <= 1.0, "Fitness out of range"
    assert tree.accuracy_ is not None, "Accuracy not set"

test_custom_fitness()
```

## Best Practices

1. **Always normalize**: Keep fitness in [0, 1] range
2. **Store components**: Set `tree.accuracy_` and `tree.interpretability_`
3. **Handle edge cases**: Check for division by zero, empty predictions
4. **Test thoroughly**: Verify fitness behaves as expected
5. **Document clearly**: Explain fitness formula and design decisions

## Next Steps

- [Custom Operators](custom-operators.md)
- [Multi-Objective Optimization](multi-objective.md)
- [Statistical Tests](statistical-tests.md)