# Medical Diagnosis Example

Healthcare application prioritizing sensitivity and interpretability.

## Scenario

**Binary classification**: Detect presence of breast cancer  
**Priority**: High recall (don't miss positive cases)  
**Requirement**: Interpretable model for clinical use

## Custom Fitness for Medical Use

```python
from ga_trees.fitness.calculator import FitnessCalculator, TreePredictor
from sklearn.metrics import recall_score, confusion_matrix

class MedicalFitness(FitnessCalculator):
    """Fitness optimized for medical diagnosis."""
    
    def calculate_fitness(self, tree, X, y):
        self.predictor.fit_leaf_predictions(tree, X, y)
        y_pred = self.predictor.predict(tree, X)
        
        # Calculate sensitivity (recall) - most important
        sensitivity = recall_score(y, y_pred, pos_label=1)
        
        # Calculate specificity
        tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        # Combined clinical metric (favor sensitivity)
        clinical_score = 0.70 * sensitivity + 0.30 * specificity
        
        # Interpretability (critical for doctor acceptance)
        interp = self.interp_calc.calculate_composite_score(
            tree, {
                'node_complexity': 0.70,  # Must be simple
                'feature_coherence': 0.15,
                'tree_balance': 0.10,
                'semantic_coherence': 0.05
            }
        )
        
        tree.accuracy_ = clinical_score
        tree.interpretability_ = interp
        
        # Strong emphasis on interpretability for medical
        return 0.60 * clinical_score + 0.40 * interp
```

## Complete Medical Example

```python
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np

# Load data
X, y = load_breast_cancer(return_X_y=True)
feature_names = load_breast_cancer().feature_names
class_names = ['benign', 'malignant']

# Split and standardize
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Setup with medical fitness
n_features = X_train.shape[1]
feature_ranges = {i: (X_train[:, i].min(), X_train[:, i].max()) 
                 for i in range(n_features)}

ga_config = GAConfig(
    population_size=60,
    n_generations=40,
    mutation_types={
        'threshold_perturbation': 0.40,
        'feature_replacement': 0.20,
        'prune_subtree': 0.35,  # Higher pruning for simplicity
        'expand_leaf': 0.05
    }
)

initializer = TreeInitializer(
    n_features=n_features,
    n_classes=2,
    max_depth=4,  # Shallow for interpretability
    min_samples_split=15,
    min_samples_leaf=8
)

medical_fitness = MedicalFitness()
mutation = Mutation(n_features=n_features, feature_ranges=feature_ranges)

# Train
ga_engine = GAEngine(ga_config, initializer, medical_fitness.calculate_fitness, mutation)
best_tree = ga_engine.evolve(X_train, y_train, verbose=True)

# Evaluate
predictor = TreePredictor()
y_pred = predictor.predict(best_tree, X_test)

# Clinical metrics
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
sensitivity = tp / (tp + fn)
specificity = tn / (tn + fp)

print("\n" + "="*60)
print("CLINICAL EVALUATION")
print("="*60)
print(f"\nSensitivity (Recall): {sensitivity:.4f}")
print(f"Specificity: {specificity:.4f}")
print(f"False Negatives: {fn} (missed malignant cases)")
print(f"False Positives: {fp} (unnecessary biopsies)")

print(f"\nTree Interpretability:")
print(f"  Depth: {best_tree.get_depth()} (shallow = easy to explain)")
print(f"  Nodes: {best_tree.get_num_nodes()} (small = easy to memorize)")
print(f"  Features: {best_tree.get_num_features_used()}/{n_features} (focused)")

# Extract diagnostic rules
rules = best_tree.to_rules()
print(f"\nDiagnostic Protocol ({len(rules)} rules):")
for i, rule in enumerate(rules, 1):
    for j, name in enumerate(feature_names):
        rule = rule.replace(f"X[{j}]", name)
    print(f"  {i}. {rule}")
```