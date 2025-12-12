


# Training Models

Complete guide to training GA-optimized decision trees.

## Overview

Training involves evolving a population of decision trees using genetic algorithms to optimize both accuracy and interpretability. The framework supports:

- **Configuration-driven training** via YAML files
- **Command-line parameter overrides**
- **Multiple datasets** (built-in and custom)
- **Flexible fitness functions**
- **Automated model saving and evaluation**

## Preparing Your Data

Before training, ensure your data is properly formatted and preprocessed. The built-in [Dataset Loader](../data/dataset-loader.md) supports:
- 15+ benchmark datasets
- CSV/Excel file loading
- Automatic validation and preprocessing
- Train/test splitting
## Quick Training Example

```bash
# Train on Iris with default settings
python scripts/train.py --dataset iris

# Train with custom config
python scripts/train.py --config configs/custom.yaml --dataset breast_cancer

# Override specific parameters
python scripts/train.py --config configs/custom.yaml --generations 60 --population 120
```

## Training Workflow

### 1. Basic Training Script

```python
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from ga_trees.ga.engine import GAEngine, GAConfig, TreeInitializer, Mutation
from ga_trees.fitness.calculator import FitnessCalculator, TreePredictor

# Load data
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Setup parameters
n_features = X_train.shape[1]
n_classes = len(np.unique(y))
feature_ranges = {i: (X_train[:, i].min(), X_train[:, i].max()) 
                 for i in range(n_features)}

# Configure GA
ga_config = GAConfig(
    population_size=80,
    n_generations=40,
    crossover_prob=0.72,
    mutation_prob=0.18,
    tournament_size=4,
    elitism_ratio=0.12
)

# Configure tree constraints
initializer = TreeInitializer(
    n_features=n_features,
    n_classes=n_classes,
    max_depth=6,
    min_samples_split=8,
    min_samples_leaf=3
)

# Configure fitness function
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

# Configure mutation
mutation = Mutation(n_features=n_features, feature_ranges=feature_ranges)

# Create GA engine
ga_engine = GAEngine(
    config=ga_config,
    initializer=initializer,
    fitness_function=fitness_calc.calculate_fitness,
    mutation=mutation
)

# Train
print("Starting evolution...")
best_tree = ga_engine.evolve(X_train, y_train, verbose=True)

# Evaluate
predictor = TreePredictor()
y_pred = predictor.predict(best_tree, X_test)

from sklearn.metrics import accuracy_score, classification_report
print(f"\nTest Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"Tree Nodes: {best_tree.get_num_nodes()}")
print(f"Tree Depth: {best_tree.get_depth()}")
```

**Expected Output:**

```
Starting evolution...
Gen 0: Best=0.8234, Avg=0.6543
Gen 10: Best=0.8756, Avg=0.7892
Gen 20: Best=0.9123, Avg=0.8456
Gen 30: Best=0.9345, Avg=0.8876
Gen 40: Best=0.9456, Avg=0.9012

Test Accuracy: 0.9333
Tree Nodes: 7
Tree Depth: 3
```

### 2. Using Configuration Files

**Create config file** (`my_config.yaml`):

```yaml
ga:
  population_size: 80
  n_generations: 40
  crossover_prob: 0.72
  mutation_prob: 0.18
  tournament_size: 4
  elitism_ratio: 0.12
  
  mutation_types:
    threshold_perturbation: 0.45
    feature_replacement: 0.25
    prune_subtree: 0.25
    expand_leaf: 0.05

tree:
  max_depth: 6
  min_samples_split: 8
  min_samples_leaf: 3

fitness:
  mode: weighted_sum
  weights:
    accuracy: 0.68
    interpretability: 0.32
  
  interpretability_weights:
    node_complexity: 0.50
    feature_coherence: 0.10
    tree_balance: 0.10
    semantic_coherence: 0.30
```

**Train with config:**

```bash
python scripts/train.py --config my_config.yaml --dataset breast_cancer
```

### 3. Custom Dataset Training

```python
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load custom dataset
df = pd.read_csv('data/my_dataset.csv')

# Prepare features and labels
X = df.drop('target', axis=1).values
y = df['target'].values

# Encode labels if categorical
if y.dtype == object:
    le = LabelEncoder()
    y = le.fit_transform(y)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Continue with training as shown above...
```

## Training Parameters

### GA Configuration (`GAConfig`)

|Parameter|Type|Default|Description|
|---|---|---|---|
|`population_size`|int|80|Number of trees in population|
|`n_generations`|int|40|Number of evolution iterations|
|`crossover_prob`|float|0.72|Probability of crossover (0-1)|
|`mutation_prob`|float|0.18|Probability of mutation (0-1)|
|`tournament_size`|int|4|Tournament selection size|
|`elitism_ratio`|float|0.12|Fraction of elite preserved (0-1)|
|`mutation_types`|dict|See below|Mutation type probabilities|

**Mutation Types (must sum to 1.0):**

```python
{
    'threshold_perturbation': 0.45,  # Adjust split thresholds
    'feature_replacement': 0.25,     # Change split features
    'prune_subtree': 0.25,           # Remove subtrees
    'expand_leaf': 0.05              # Grow trees
}
```

### Tree Constraints (`TreeInitializer`)

|Parameter|Type|Default|Description|
|---|---|---|---|
|`max_depth`|int|6|Maximum tree depth|
|`min_samples_split`|int|8|Min samples to split node|
|`min_samples_leaf`|int|3|Min samples in leaf|

**Constraint Impact:**

- **Stricter constraints** → Smaller, more interpretable trees
- **Looser constraints** → Larger, potentially more accurate trees

### Fitness Configuration (`FitnessCalculator`)

```python
FitnessCalculator(
    mode='weighted_sum',              # 'weighted_sum' or 'pareto'
    accuracy_weight=0.68,             # Weight for accuracy
    interpretability_weight=0.32,     # Weight for interpretability
    interpretability_weights={
        'node_complexity': 0.50,      # Penalty for tree size
        'feature_coherence': 0.10,    # Reward feature reuse
        'tree_balance': 0.10,         # Reward balanced trees
        'semantic_coherence': 0.30    # Reward prediction consistency
    }
)
```

**Fitness Mode:**

- `weighted_sum`: Single objective = accuracy_weight × acc + interpretability_weight × interp
- `pareto`: Multi-objective optimization (experimental)




## Advanced Training Options

### 1. Early Stopping

```python
class EarlyStoppingGA(GAEngine):
    def __init__(self, *args, patience=10, min_delta=0.001, **kwargs):
        super().__init__(*args, **kwargs)
        self.patience = patience
        self.min_delta = min_delta
        self.best_fitness = -np.inf
        self.patience_counter = 0
    
    def evolve(self, X, y, verbose=True):
        for generation in range(self.config.n_generations):
            # ... standard evolution ...
            
            current_best = max(ind.fitness_ for ind in self.population)
            
            # Check for improvement
            if current_best > self.best_fitness + self.min_delta:
                self.best_fitness = current_best
                self.patience_counter = 0
            else:
                self.patience_counter += 1
            
            # Early stop
            if self.patience_counter >= self.patience:
                print(f"Early stopping at generation {generation}")
                break
        
        return self.best_individual
```

### 2. Custom Fitness Function

```python
def medical_fitness(tree, X, y):
    """Custom fitness for medical diagnosis: prioritize recall."""
    from sklearn.metrics import recall_score
    
    # Fit predictions
    predictor = TreePredictor()
    predictor.fit_leaf_predictions(tree, X, y)
    y_pred = predictor.predict(tree, X)
    
    # Calculate recall (sensitivity)
    recall = recall_score(y, y_pred, average='weighted')
    
    # Interpretability
    interp = 1.0 / (1.0 + tree.get_num_nodes() / 20.0)
    
    # Weighted fitness (80% recall, 20% interpretability)
    return 0.80 * recall + 0.20 * interp

# Use custom fitness
ga_engine = GAEngine(
    config=ga_config,
    initializer=initializer,
    fitness_function=medical_fitness,  # Custom function
    mutation=mutation
)
```

### 3. Cross-Validation During Training

```python
from sklearn.model_selection import StratifiedKFold

def cv_fitness(tree, X, y, n_folds=3):
    """Fitness with cross-validation."""
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    predictor = TreePredictor()
    scores = []
    
    for train_idx, val_idx in skf.split(X, y):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        predictor.fit_leaf_predictions(tree, X_train, y_train)
        y_pred = predictor.predict(tree, X_val)
        scores.append(accuracy_score(y_val, y_pred))
    
    accuracy = np.mean(scores)
    interp = tree.interpretability_
    
    return 0.7 * accuracy + 0.3 * interp
```

### 4. Warm Start from Existing Model

```python
import pickle

# Load previous best model
with open('models/previous_best.pkl', 'rb') as f:
    prev_model = pickle.load(f)
    prev_tree = prev_model['tree']

# Initialize population with previous best
ga_engine.initialize_population(X_train, y_train)
ga_engine.population[0] = prev_tree.copy()  # Seed with previous best

# Continue evolution
best_tree = ga_engine.evolve(X_train, y_train, verbose=True)
```

## Model Evaluation

### Complete Evaluation Pipeline

```python
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt

# Predict
predictor = TreePredictor()
y_train_pred = predictor.predict(best_tree, X_train)
y_test_pred = predictor.predict(best_tree, X_test)

# Metrics
print("="*60)
print("MODEL EVALUATION")
print("="*60)

print(f"\nTrain Accuracy: {accuracy_score(y_train, y_train_pred):.4f}")
print(f"Test Accuracy:  {accuracy_score(y_test, y_test_pred):.4f}")
print(f"Test F1 Score:  {f1_score(y_test, y_test_pred, average='weighted'):.4f}")

print(f"\nTree Statistics:")
print(f"  Depth: {best_tree.get_depth()}")
print(f"  Nodes: {best_tree.get_num_nodes()}")
print(f"  Leaves: {best_tree.get_num_leaves()}")
print(f"  Features Used: {best_tree.get_num_features_used()}/{n_features}")
print(f"  Balance: {best_tree.get_tree_balance():.4f}")

print(f"\nFitness Components:")
print(f"  Overall Fitness: {best_tree.fitness_:.4f}")
print(f"  Accuracy: {best_tree.accuracy_:.4f}")
print(f"  Interpretability: {best_tree.interpretability_:.4f}")

# Classification report
print("\nClassification Report:")
print(classification_report(y_test, y_test_pred))

# Confusion matrix
cm = confusion_matrix(y_test, y_test_pred)
print("\nConfusion Matrix:")
print(cm)

# Plot confusion matrix
import seaborn as sns
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.savefig('results/confusion_matrix.png')
```

## Saving and Loading Models

### Save Model

```python
import pickle
from pathlib import Path

# Prepare model data
model_data = {
    'tree': best_tree,
    'scaler': scaler,
    'feature_ranges': feature_ranges,
    'n_features': n_features,
    'n_classes': n_classes,
    'config': vars(ga_config),
    'metrics': {
        'train_accuracy': accuracy_score(y_train, y_train_pred),
        'test_accuracy': accuracy_score(y_test, y_test_pred),
        'test_f1': f1_score(y_test, y_test_pred, average='weighted'),
    }
}

# Save
output_path = Path('models/best_tree.pkl')
output_path.parent.mkdir(parents=True, exist_ok=True)

with open(output_path, 'wb') as f:
    pickle.dump(model_data, f)

print(f"✓ Model saved to: {output_path}")
```

### Load Model

```python
# Load model
with open('models/best_tree.pkl', 'rb') as f:
    model_data = pickle.load(f)

loaded_tree = model_data['tree']
loaded_scaler = model_data['scaler']

# Use for prediction
X_new = loaded_scaler.transform(X_new_raw)
y_pred = predictor.predict(loaded_tree, X_new)
```

## Monitoring Training Progress

### Evolution History

```python
# After training
history = ga_engine.get_history()

import matplotlib.pyplot as plt

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

# Fitness evolution
ax1.plot(history['best_fitness'], label='Best', linewidth=2)
ax1.plot(history['avg_fitness'], label='Average', linewidth=2, alpha=0.7)
ax1.set_xlabel('Generation')
ax1.set_ylabel('Fitness')
ax1.set_title('Fitness Evolution')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Diversity (if tracked)
if 'diversity' in history:
    ax2.plot(history['diversity'], linewidth=2, color='green')
    ax2.set_xlabel('Generation')
    ax2.set_ylabel('Population Diversity')
    ax2.set_title('Population Diversity')
    ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/evolution_history.png')
```

### Real-Time Progress

```python
# Custom callback for detailed progress
class TrainingCallback:
    def __init__(self):
        self.generation_stats = []
    
    def on_generation_end(self, generation, population, best_individual):
        stats = {
            'generation': generation,
            'best_fitness': best_individual.fitness_,
            'avg_fitness': np.mean([ind.fitness_ for ind in population]),
            'best_depth': best_individual.get_depth(),
            'best_nodes': best_individual.get_num_nodes(),
        }
        self.generation_stats.append(stats)
        
        # Print progress
        if generation % 5 == 0:
            print(f"Gen {generation:3d}: "
                  f"Fitness={stats['best_fitness']:.4f}, "
                  f"Nodes={stats['best_nodes']:2d}, "
                  f"Depth={stats['best_depth']}")

# Integrate into GA engine (requires modification)
```

## Tips for Effective Training

### 1. Start Small, Scale Up

```bash
# Quick test (1-2 minutes)
python scripts/train.py --dataset iris --population 30 --generations 10

# Medium run (5-10 minutes)
python scripts/train.py --dataset wine --population 50 --generations 30

# Full run (30+ minutes)
python scripts/train.py --config configs/custom.yaml --dataset breast_cancer
```

### 2. Balance Accuracy vs Interpretability

|Use Case|Accuracy Weight|Interpretability Weight|
|---|---|---|
|Medical diagnosis|0.85|0.15|
|Regulatory compliance|0.50|0.50|
|Exploratory analysis|0.60|0.40|
|Production model|0.75|0.25|

### 3. Hyperparameter Sensitivity

**Most impactful parameters:**

1. `population_size` (30-150)
2. `n_generations` (20-100)
3. `accuracy_weight` (0.5-0.9)
4. `max_depth` (4-8)

**Less sensitive:**

- `crossover_prob` (0.6-0.8)
- `mutation_prob` (0.1-0.3)

### 4. When to Use Each Mutation Type

- **`threshold_perturbation`** (40-50%): Fine-tune decision boundaries
- **`feature_replacement`** (20-30%): Explore alternative features
- **`prune_subtree`** (20-30%): Simplify overgrown trees
- **`expand_leaf`** (5-10%): Add complexity when needed

## Troubleshooting

### Issue: Fitness Not Improving

**Solutions:**

1. Increase population size (50 → 100)
2. Increase generations (30 → 60)
3. Adjust mutation probability (0.18 → 0.25)
4. Check feature scaling (standardize data)

### Issue: Trees Too Large

**Solutions:**

1. Increase interpretability weight (0.32 → 0.50)
2. Increase node_complexity weight (0.50 → 0.70)
3. Decrease max_depth (6 → 4)
4. Increase prune_subtree mutation (0.25 → 0.40)

### Issue: Overfitting

**Solutions:**

1. Use cross-validation fitness
2. Increase min_samples_leaf (3 → 5)
3. Regularize with interpretability weight
4. Add early stopping

## Next Steps

- **Run Experiments**: See Experiments Guide for benchmarking
- **Hyperparameter Tuning**: Use Optuna Guide for optimization
- **Visualization**: Learn to visualize results
- **Custom Operators**: Create custom genetic operators