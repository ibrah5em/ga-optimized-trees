# Model Export and Loading

Save and load trained models.

## Saving Models

```python
import pickle
from pathlib import Path

# Prepare model data
model_data = {
    'tree': best_tree,
    'scaler': scaler,  # If used
    'feature_ranges': feature_ranges,
    'feature_names': feature_names,
    'class_names': class_names,
    'n_features': n_features,
    'n_classes': n_classes,
    'config': vars(ga_config),
    'metrics': {
        'train_accuracy': train_acc,
        'test_accuracy': test_acc,
        'test_f1': test_f1,
        'nodes': best_tree.get_num_nodes(),
        'depth': best_tree.get_depth()
    }
}

# Save
output_path = Path('models/best_tree.pkl')
output_path.parent.mkdir(parents=True, exist_ok=True)
with open(output_path, 'wb') as f:
    pickle.dump(model_data, f)

print(f"Model saved to: {output_path}")
```

## Loading Models

```python
import pickle

# Load
with open('models/best_tree.pkl', 'rb') as f:
    model_data = pickle.load(f)

# Extract components
tree = model_data['tree']
scaler = model_data.get('scaler')
feature_names = model_data.get('feature_names')
class_names = model_data.get('class_names')
metrics = model_data.get('metrics', {})

print(f"Loaded model:")
print(f"  Nodes: {metrics.get('nodes')}")
print(f"  Accuracy: {metrics.get('test_accuracy'):.4f}")
```

## Making Predictions

```python
from ga_trees.fitness.calculator import TreePredictor

# Prepare new data
X_new = load_new_data()
if scaler:
    X_new = scaler.transform(X_new)

# Predict
predictor = TreePredictor()
y_pred = predictor.predict(tree, X_new)

print(f"Predictions: {y_pred}")
```