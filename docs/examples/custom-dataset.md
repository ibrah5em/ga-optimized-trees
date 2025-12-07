# Custom Dataset Example

Using your own data with the framework.

## Loading Custom Data

### From CSV

```python
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load data
df = pd.read_csv('my_dataset.csv')

# Separate features and target
X = df.drop('target_column', axis=1).values
y = df['target_column'].values

# Encode categorical target
if y.dtype == object:
    le = LabelEncoder()
    y = le.fit_transform(y)
    class_names = le.classes_
else:
    class_names = None

print(f"Loaded: {X.shape[0]} samples, {X.shape[1]} features")
```

### From NumPy Arrays

```python
import numpy as np

# Your data arrays
X = np.load('features.npy')
y = np.load('labels.npy')

# Verify shapes
assert X.ndim == 2, "X must be 2D (samples × features)"
assert y.ndim == 1, "y must be 1D (samples,)"
assert X.shape[0] == y.shape[0], "X and y must have same number of samples"
```

## Complete Pipeline

```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from ga_trees.ga.engine import GAEngine, GAConfig, TreeInitializer, Mutation
from ga_trees.fitness.calculator import FitnessCalculator

# 1. Load your data
X, y = load_your_data()  # Your loading function
feature_names = ['feature_' + str(i) for i in range(X.shape[1])]

# 2. Preprocessing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 3. Setup GA components
n_features = X_train.shape[1]
n_classes = len(np.unique(y))
feature_ranges = {i: (X_train[:, i].min(), X_train[:, i].max()) 
                 for i in range(n_features)}

# 4. Configure for your problem
ga_config = GAConfig(
    population_size=80,     # Adjust based on problem complexity
    n_generations=40,       # Adjust based on time budget
    crossover_prob=0.72,
    mutation_prob=0.18
)

initializer = TreeInitializer(
    n_features=n_features,
    n_classes=n_classes,
    max_depth=6,           # Adjust based on desired interpretability
    min_samples_split=10,
    min_samples_leaf=5
)

fitness_calc = FitnessCalculator(
    accuracy_weight=0.68,         # Tune for your needs
    interpretability_weight=0.32
)

mutation = Mutation(n_features=n_features, feature_ranges=feature_ranges)

# 5. Train
ga_engine = GAEngine(ga_config, initializer, fitness_calc.calculate_fitness, mutation)
best_tree = ga_engine.evolve(X_train, y_train, verbose=True)

# 6. Evaluate
from ga_trees.fitness.calculator import TreePredictor
from sklearn.metrics import accuracy_score, classification_report

predictor = TreePredictor()
y_pred = predictor.predict(best_tree, X_test)

print(f"\nAccuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"Nodes: {best_tree.get_num_nodes()}")
print(classification_report(y_test, y_pred))
```
