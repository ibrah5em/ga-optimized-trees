# Tree Representation

Understanding the tree genotype structure.

## Overview

Trees are represented as **binary decision trees** with:

- **Internal nodes**: Decision points (feature, threshold, operator)
- **Leaf nodes**: Predictions (class label or regression value)

## Node Structure

### Internal Node

```python
Node(
    node_type="internal",
    feature_idx=0,  # Feature to split on
    threshold=0.5,  # Split threshold
    operator="<=",  # Comparison operator
    left_child=Node(...),  # True branch (<=)
    right_child=Node(...),  # False branch (>)
    depth=0,  # Depth in tree
)
```

### Leaf Node

```python
Node(node_type="leaf", prediction=1, depth=2)  # Class label or value
```

## Tree Example

**Simple 3-node tree:**

```
         [X[0] <= 0.5]
         /           \
    class=0        class=1
```

**Code:**

```python
from ga_trees.genotype.tree_genotype import (
    create_internal_node,
    create_leaf_node,
    TreeGenotype,
)

left = create_leaf_node(prediction=0, depth=1)
right = create_leaf_node(prediction=1, depth=1)
root = create_internal_node(
    feature_idx=0, threshold=0.5, left_child=left, right_child=right, depth=0
)

tree = TreeGenotype(
    root=root,
    n_features=4,
    n_classes=2,
    max_depth=5,
    min_samples_split=10,
    min_samples_leaf=5,
)
```

## Tree Constraints

All trees must satisfy:

1. **Max Depth**: `depth ≤ max_depth`
1. **Min Samples Split**: `samples_at_node ≥ min_samples_split`
1. **Min Samples Leaf**: `samples_at_leaf ≥ min_samples_leaf`
1. **Valid Features**: `0 ≤ feature_idx < n_features`
1. **Binary Structure**: Each internal node has exactly 2 children

## Tree Operations

### Traversal

**Making a prediction:**

```python
def predict_single(node, x):
    if node.is_leaf():
        return node.prediction

    if x[node.feature_idx] <= node.threshold:
        return predict_single(node.left_child, x)
    else:
        return predict_single(node.right_child, x)
```

### Inspection

```python
# Get tree statistics
depth = tree.get_depth()  # Maximum depth
nodes = tree.get_num_nodes()  # Total nodes
leaves = tree.get_num_leaves()  # Leaf count
features = tree.get_features_used()  # Set of features used

# Get all nodes
all_nodes = tree.get_all_nodes()
internal_nodes = tree.get_internal_nodes()
leaf_nodes = tree.get_all_leaves()

# Tree balance
balance = tree.get_tree_balance()  # [0, 1], higher = more balanced
```

### Validation

```python
valid, errors = tree.validate()
if not valid:
    print("Tree validation failed:")
    for error in errors:
        print(f"  - {error}")
```

### Rule Extraction

```python
rules = tree.to_rules()
for i, rule in enumerate(rules, 1):
    print(f"Rule {i}: {rule}")

# Output:
# Rule 1: IF X[0] <= 0.5000 AND X[1] <= 0.3000 THEN class=0
# Rule 2: IF X[0] <= 0.5000 AND X[1] > 0.3000 THEN class=1
# Rule 3: IF X[0] > 0.5000 THEN class=2
```

## Genotype vs Phenotype

**Genotype** (structure):

- Tree topology (node connections)
- Split features and thresholds
- Internal representation

**Phenotype** (behavior):

- Predictions on data
- Decision boundaries
- Observable performance

**Evolution modifies genotype** → **Fitness evaluates phenotype**

## Memory and Copying

Trees use **deep copying** to prevent interference:

```python
# Always copy before modification
tree_copy = tree.copy()
mutated_tree = mutation.mutate(tree_copy)

# Original tree unchanged
assert tree.get_num_nodes() == original_nodes
```
