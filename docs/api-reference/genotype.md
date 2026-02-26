# Genotype Module API Reference

The genotype module defines the tree structure representation used by the genetic algorithm.

## Module: `ga_trees.genotype.tree_genotype`

### Classes

## `Node`

Represents a single node in the decision tree (internal or leaf).

### Attributes

| Attribute       | Type                                      | Description                                    |
| --------------- | ----------------------------------------- | ---------------------------------------------- |
| `node_type`     | `Literal['internal', 'leaf']`             | Type of node                                   |
| `feature_idx`   | `Optional[int]`                           | Feature index for splits (internal nodes only) |
| `threshold`     | `Optional[float]`                         | Split threshold (internal nodes only)          |
| `operator`      | `Literal['<=', '>']`                      | Comparison operator (default: '\<=')           |
| `left_child`    | `Optional[Node]`                          | Left child node                                |
| `right_child`   | `Optional[Node]`                          | Right child node                               |
| `prediction`    | `Optional[Union[int, float, np.ndarray]]` | Prediction value (leaf nodes only)             |
| `depth`         | `int`                                     | Depth in tree (root = 0)                       |
| `node_id`       | `int`                                     | Unique node identifier                         |
| `samples_count` | `int`                                     | Number of samples reaching this node           |

### Methods

#### `is_leaf() -> bool`

Check if this is a leaf node.

**Returns:** `True` if leaf, `False` otherwise

**Example:**

```python
if node.is_leaf():
    print(f"Prediction: {node.prediction}")
```

#### `is_internal() -> bool`

Check if this is an internal node.

**Returns:** `True` if internal, `False` otherwise

#### `get_height() -> int`

Get the height of subtree rooted at this node.

**Returns:** Height (leaf = 0)

**Example:**

```python
height = node.get_height()
print(f"Subtree height: {height}")
```

#### `count_nodes() -> int`

Count total nodes in subtree.

**Returns:** Number of nodes (including this node)

#### `get_leaf_depths() -> List[int]`

Get depths of all leaves in subtree.

**Returns:** List of leaf depths

**Example:**

```python
depths = node.get_leaf_depths()
print(f"Leaf depths: {depths}")
print(f"Avg depth: {np.mean(depths):.2f}")
```

#### `get_features_used() -> set`

Get set of features used in subtree.

**Returns:** Set of feature indices

**Example:**

```python
features = node.get_features_used()
print(f"Features used: {features}")
```

#### `copy() -> Node`

Create a deep copy of this node and its subtree.

**Returns:** New `Node` instance

______________________________________________________________________

## `TreeGenotype`

Main genotype representation of a decision tree.

### Constructor

```python
TreeGenotype(
    root: Node,
    n_features: int,
    n_classes: int,
    task_type: Literal['classification', 'regression'] = 'classification',
    max_depth: int = 5,
    min_samples_split: int = 10,
    min_samples_leaf: int = 5,
    max_features: Optional[int] = None
)
```

**Parameters:**

- `root`: Root node of the tree
- `n_features`: Number of input features
- `n_classes`: Number of target classes
- `task_type`: Task type ('classification' or 'regression')
- `max_depth`: Maximum allowed tree depth
- `min_samples_split`: Minimum samples required to split
- `min_samples_leaf`: Minimum samples required in leaf
- `max_features`: Maximum features to consider (None = all)

**Example:**

```python
from ga_trees.genotype.tree_genotype import (
    TreeGenotype,
    create_leaf_node,
    create_internal_node,
)

# Create simple tree
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

### Attributes

| Attribute           | Type              | Description                                    |
| ------------------- | ----------------- | ---------------------------------------------- |
| `root`              | `Node`            | Root node of the tree                          |
| `n_features`        | `int`             | Number of features                             |
| `n_classes`         | `int`             | Number of classes                              |
| `task_type`         | `str`             | 'classification' or 'regression'               |
| `max_depth`         | `int`             | Maximum depth constraint                       |
| `min_samples_split` | `int`             | Minimum samples to split                       |
| `min_samples_leaf`  | `int`             | Minimum samples in leaf                        |
| `fitness_`          | `Optional[float]` | Fitness score (set during evolution)           |
| `accuracy_`         | `Optional[float]` | Accuracy score (set during evaluation)         |
| `interpretability_` | `Optional[float]` | Interpretability score (set during evaluation) |

### Methods

#### Structure Methods

##### `get_depth() -> int`

Get maximum depth of the tree.

**Returns:** Tree depth

**Example:**

```python
depth = tree.get_depth()
print(f"Tree depth: {depth}")
```

##### `get_num_nodes() -> int`

Get total number of nodes.

**Returns:** Node count

##### `get_num_leaves() -> int`

Get number of leaf nodes.

**Returns:** Leaf count

##### `get_all_nodes() -> List[Node]`

Get list of all nodes in tree (breadth-first traversal).

**Returns:** List of all nodes

**Example:**

```python
nodes = tree.get_all_nodes()
print(f"Total nodes: {len(nodes)}")

for node in nodes:
    if node.is_internal():
        print(f"Node {node.node_id}: split on feature {node.feature_idx}")
```

##### `get_all_leaves() -> List[Node]`

Get list of all leaf nodes.

**Returns:** List of leaf nodes

##### `get_internal_nodes() -> List[Node]`

Get list of all internal nodes.

**Returns:** List of internal nodes

#### Feature Analysis

##### `get_features_used() -> set`

Get set of all features used in tree.

**Returns:** Set of feature indices

**Example:**

```python
features = tree.get_features_used()
print(f"Features used: {features}")
print(f"Feature count: {len(features)}/{tree.n_features}")
```

##### `get_num_features_used() -> int`

Get count of unique features used.

**Returns:** Number of unique features

#### Interpretability Metrics

##### `get_tree_balance() -> float`

Calculate tree balance metric.

**Returns:** Balance score in \[0, 1\] where 1 is perfectly balanced

**Formula:**

```
balance = 1 - min(std(leaf_depths) / max_depth, 1.0)
```

**Example:**

```python
balance = tree.get_tree_balance()
print(f"Tree balance: {balance:.4f}")
if balance > 0.8:
    print("Tree is well-balanced")
```

#### Validation

##### `validate() -> Tuple[bool, List[str]]`

Validate tree structure and constraints.

**Returns:** Tuple of (is_valid, list_of_errors)

**Checks:**

- Depth constraint
- Feature indices validity
- Threshold presence in internal nodes
- Prediction presence in leaf nodes
- Tree structure consistency

**Example:**

```python
valid, errors = tree.validate()
if not valid:
    print("Tree validation failed:")
    for error in errors:
        print(f"  - {error}")
else:
    print("Tree is valid")
```

#### Utility Methods

##### `copy() -> TreeGenotype`

Create a deep copy of this tree.

**Returns:** New `TreeGenotype` instance

**Example:**

```python
tree_copy = tree.copy()
tree_copy.root.threshold = 0.3  # Modify copy
assert tree.root.threshold != 0.3  # Original unchanged
```

##### `to_dict() -> dict`

Convert tree to dictionary representation.

**Returns:** Dictionary with tree structure and metadata

**Example:**

```python
tree_dict = tree.to_dict()
print(f"Tree depth: {tree_dict['metadata']['depth']}")
print(f"Tree nodes: {tree_dict['metadata']['num_nodes']}")
```

##### `to_rules() -> List[str]`

Extract human-readable rules from tree.

**Returns:** List of rule strings

**Example:**

```python
rules = tree.to_rules()
print("Decision Rules:")
for i, rule in enumerate(rules, 1):
    print(f"{i}. {rule}")

# Output:
# 1. IF X[0] <= 0.5000 AND X[1] <= 0.3000 THEN class=0
# 2. IF X[0] <= 0.5000 AND X[1] > 0.3000 THEN class=1
# 3. IF X[0] > 0.5000 THEN class=1
```

______________________________________________________________________

## Factory Functions

### `create_leaf_node()`

Create a leaf node.

```python
create_leaf_node(
    prediction: Union[int, float, np.ndarray],
    depth: int = 0
) -> Node
```

**Parameters:**

- `prediction`: Prediction value
- `depth`: Node depth (default: 0)

**Returns:** Leaf `Node`

**Example:**

```python
leaf = create_leaf_node(prediction=1, depth=2)
assert leaf.is_leaf()
assert leaf.prediction == 1
```

### `create_internal_node()`

Create an internal node.

```python
create_internal_node(
    feature_idx: int,
    threshold: float,
    left_child: Node,
    right_child: Node,
    depth: int = 0
) -> Node
```

**Parameters:**

- `feature_idx`: Feature index for split
- `threshold`: Split threshold
- `left_child`: Left child node (≤ threshold)
- `right_child`: Right child node (> threshold)
- `depth`: Node depth (default: 0)

**Returns:** Internal `Node`

**Example:**

```python
left = create_leaf_node(0, depth=1)
right = create_leaf_node(1, depth=1)
internal = create_internal_node(
    feature_idx=0, threshold=0.5, left_child=left, right_child=right, depth=0
)
assert internal.is_internal()
assert internal.feature_idx == 0
```

______________________________________________________________________

## Usage Examples

### Example 1: Create Custom Tree

```python
from ga_trees.genotype.tree_genotype import (
    TreeGenotype,
    create_leaf_node,
    create_internal_node,
)

# Build tree manually
# Tree structure:
#       [X[0] <= 0.5]
#       /           \
#   [X[1] <= 0.3]   class=2
#    /         \
# class=0    class=1

leaf0 = create_leaf_node(0, depth=2)
leaf1 = create_leaf_node(1, depth=2)
leaf2 = create_leaf_node(2, depth=1)

left_branch = create_internal_node(
    feature_idx=1, threshold=0.3, left_child=leaf0, right_child=leaf1, depth=1
)

root = create_internal_node(
    feature_idx=0, threshold=0.5, left_child=left_branch, right_child=leaf2, depth=0
)

tree = TreeGenotype(
    root=root,
    n_features=4,
    n_classes=3,
    max_depth=5,
    min_samples_split=10,
    min_samples_leaf=5,
)

# Analyze tree
print(f"Depth: {tree.get_depth()}")  # Output: 2
print(f"Nodes: {tree.get_num_nodes()}")  # Output: 5
print(f"Leaves: {tree.get_num_leaves()}")  # Output: 3
print(f"Features: {tree.get_features_used()}")  # Output: {0, 1}
```

### Example 2: Tree Inspection

```python
# Get all internal nodes
for node in tree.get_internal_nodes():
    print(f"Split on feature {node.feature_idx} at {node.threshold:.4f}")

# Get leaf predictions
for leaf in tree.get_all_leaves():
    print(f"Leaf at depth {leaf.depth}: predicts {leaf.prediction}")

# Extract rules
rules = tree.to_rules()
for rule in rules:
    print(rule)
```

### Example 3: Tree Modification

```python
# Copy tree for modification
modified_tree = tree.copy()

# Change a threshold
internal_nodes = modified_tree.get_internal_nodes()
internal_nodes[0].threshold = 0.6

# Validate modified tree
valid, errors = modified_tree.validate()
if not valid:
    print(f"Validation errors: {errors}")
```

## See Also

- [GA Engine API](ga-engine.md)
- [Fitness Calculator API](fitness.md)
- [Tree Representation Guide](../core-concepts/tree-representation.md)
