# Visualization Guide

Complete guide to visualizing GA-optimized decision trees, experiment results, and evolution dynamics.

## Overview

The framework provides visualization tools for:

- **Tree structures** (graphical representation)
- **Evolution history** (fitness over generations)
- **Comparative results** (GA vs baselines)
- **Statistical analysis** (p-values, effect sizes)
- **Feature importance**
- **Pareto fronts** (accuracy-interpretability trade-offs)

## Tree Visualization

### 1. Graphviz Tree Plot

Visualize the tree structure with splits and predictions:

```python
from ga_trees.evaluation.tree_visualizer import TreeVisualizer

# Load trained tree
import pickle
with open('models/best_tree.pkl', 'rb') as f:
    model_data = pickle.load(f)
    best_tree = model_data['tree']

# Define feature names
feature_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
class_names = ['setosa', 'versicolor', 'virginica']

# Visualize
TreeVisualizer.visualize_tree(
    best_tree,
    feature_names=feature_names,
    class_names=class_names,
    save_path='results/figures/tree_structure'
)
```

**Output:** `results/figures/tree_structure.png`

*Tree structure visualization is generated at runtime and saved to `results/figures/tree_structure.png`.*

**Interpretation:**

- **Blue boxes**: Internal nodes (decision splits)
- **Green boxes**: Leaf nodes (predictions)
- **Edges**: Decision paths (True/False)

### 2. Simple Text-Based Tree

Quick visualization without Graphviz:

```python
def print_tree(node, feature_names=None, indent=""):
    """Print tree in text format."""
    if node.is_leaf():
        print(f"{indent}→ Predict: {node.prediction}")
    else:
        feat_name = feature_names[node.feature_idx] if feature_names else f"X[{node.feature_idx}]"
        print(f"{indent}{feat_name} <= {node.threshold:.3f}")
        print(f"{indent}├─ True:")
        print_tree(node.left_child, feature_names, indent + "│  ")
        print(f"{indent}└─ False:")
        print_tree(node.right_child, feature_names, indent + "   ")

# Use it
print_tree(best_tree.root, feature_names)
```

**Output:**

```
petal_length <= 2.450
├─ True:
│  → Predict: 0
└─ False:
   petal_width <= 1.750
   ├─ True:
   │  → Predict: 1
   └─ False:
      → Predict: 2
```

### 3. Decision Rules Visualization

Extract and display rules:

```python
rules = best_tree.to_rules()

print("="*60)
print("DECISION RULES")
print("="*60)

for i, rule in enumerate(rules, 1):
    # Replace feature indices with names
    readable_rule = rule
    for j, name in enumerate(feature_names):
        readable_rule = readable_rule.replace(f"X[{j}]", name)
    
    print(f"\nRule {i}:")
    print(f"  {readable_rule}")
```

**Output:**

```
============================================================
DECISION RULES
============================================================

Rule 1:
  IF petal_length <= 2.4500 THEN class=0

Rule 2:
  IF petal_length > 2.4500 AND petal_width <= 1.7500 THEN class=1

Rule 3:
  IF petal_length > 2.4500 AND petal_width > 1.7500 THEN class=2
```

## Evolution Visualization

### 1. Fitness Over Generations

Track fitness improvement during evolution:

```python
import matplotlib.pyplot as plt
import numpy as np

# Get history from GA engine
history = ga_engine.get_history()

fig, ax = plt.subplots(figsize=(12, 6))

# Plot best and average fitness
generations = range(len(history['best_fitness']))
ax.plot(generations, history['best_fitness'], 
        label='Best Fitness', linewidth=2.5, color='#2ecc71')
ax.plot(generations, history['avg_fitness'], 
        label='Average Fitness', linewidth=2, color='#3498db', alpha=0.7)

# Fill area between
ax.fill_between(generations, history['best_fitness'], history['avg_fitness'],
                alpha=0.2, color='#3498db')

ax.set_xlabel('Generation', fontsize=12, fontweight='bold')
ax.set_ylabel('Fitness', fontsize=12, fontweight='bold')
ax.set_title('GA Evolution: Fitness Over Time', fontsize=14, fontweight='bold')
ax.legend(fontsize=11, loc='lower right')
ax.grid(True, alpha=0.3, linestyle='--')

plt.tight_layout()
plt.savefig('results/figures/fitness_evolution.png', dpi=300, bbox_inches='tight')
plt.show()
```

### 2. Multi-Metric Evolution

Track multiple metrics simultaneously:

```python
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Fitness
axes[0, 0].plot(history['best_fitness'], linewidth=2)
axes[0, 0].set_title('Best Fitness', fontweight='bold')
axes[0, 0].set_xlabel('Generation')
axes[0, 0].grid(True, alpha=0.3)

# Accuracy (if tracked)
if 'best_accuracy' in history:
    axes[0, 1].plot(history['best_accuracy'], linewidth=2, color='#e74c3c')
    axes[0, 1].set_title('Best Accuracy', fontweight='bold')
    axes[0, 1].set_xlabel('Generation')
    axes[0, 1].grid(True, alpha=0.3)

# Tree Size
if 'best_nodes' in history:
    axes[1, 0].plot(history['best_nodes'], linewidth=2, color='#f39c12')
    axes[1, 0].set_title('Tree Size (Nodes)', fontweight='bold')
    axes[1, 0].set_xlabel('Generation')
    axes[1, 0].grid(True, alpha=0.3)

# Diversity
if 'diversity' in history:
    axes[1, 1].plot(history['diversity'], linewidth=2, color='#9b59b6')
    axes[1, 1].set_title('Population Diversity', fontweight='bold')
    axes[1, 1].set_xlabel('Generation')
    axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/figures/multi_metric_evolution.png', dpi=300)
```

## Experiment Results Visualization

### 1. Accuracy Comparison Bar Chart

Compare GA against baselines:

```python
import seaborn as sns

# Results from experiment
results = {
    'iris': {'GA': 94.55, 'CART': 92.41, 'RF': 95.33},
    'wine': {'GA': 88.19, 'CART': 87.22, 'RF': 97.75},
    'breast_cancer': {'GA': 91.05, 'CART': 91.57, 'RF': 95.08}
}

# Convert to DataFrame
data = []
for dataset, models in results.items():
    for model, acc in models.items():
        data.append({'Dataset': dataset, 'Model': model, 'Accuracy': acc})

df = pd.DataFrame(data)

# Plot
fig, ax = plt.subplots(figsize=(12, 6))

sns.barplot(data=df, x='Dataset', y='Accuracy', hue='Model', 
            palette={'GA': '#2ecc71', 'CART': '#3498db', 'RF': '#e74c3c'},
            ax=ax)

ax.set_ylabel('Test Accuracy (%)', fontsize=12, fontweight='bold')
ax.set_xlabel('Dataset', fontsize=12, fontweight='bold')
ax.set_title('Model Accuracy Comparison', fontsize=14, fontweight='bold', pad=20)
ax.legend(title='Model', fontsize=11)
ax.grid(axis='y', alpha=0.3, linestyle='--')

# Add value labels on bars
for container in ax.containers:
    ax.bar_label(container, fmt='%.1f', fontsize=9)

plt.tight_layout()
plt.savefig('results/figures/accuracy_comparison.png', dpi=300)
```

### 2. Tree Size Comparison (The Key Result!)

Highlight size reduction:

```python
# Tree sizes
ga_nodes = [7.4, 10.7, 6.5]
cart_nodes = [16.4, 20.7, 35.5]
datasets = ['Iris', 'Wine', 'Breast Cancer']
reductions = [55, 48, 82]

fig, ax = plt.subplots(figsize=(12, 7))

x = np.arange(len(datasets))
width = 0.35

bars1 = ax.bar(x - width/2, ga_nodes, width, label='GA-Optimized',
               color='#2ecc71', edgecolor='black', linewidth=1.5)
bars2 = ax.bar(x + width/2, cart_nodes, width, label='CART',
               color='#e74c3c', alpha=0.85, edgecolor='black', linewidth=1.5)

# Add reduction percentages
for i, (ga, cart, red) in enumerate(zip(ga_nodes, cart_nodes, reductions)):
    y_pos = max(ga, cart) + 2
    ax.text(i, y_pos, f'{red}%↓', ha='center', fontsize=12, 
            fontweight='bold', color='#27ae60')

ax.set_ylabel('Number of Nodes', fontsize=12, fontweight='bold')
ax.set_xlabel('Dataset', fontsize=12, fontweight='bold')
ax.set_title('GA Achieves 46-82% Tree Size Reduction', 
             fontsize=14, fontweight='bold', pad=15)
ax.set_xticks(x)
ax.set_xticklabels(datasets, fontsize=11)
ax.legend(loc='upper left', fontsize=11)
ax.grid(axis='y', alpha=0.3, linestyle='--')

# Add value labels
for bars in [bars1, bars2]:
    ax.bar_label(bars, fmt='%.1f', fontsize=10)

plt.tight_layout()
plt.savefig('results/figures/tree_size_comparison.png', dpi=300)
```

### 3. Statistical Significance Visualization

Show p-values and effect sizes:

```python
# Statistical test results
datasets = ['Iris', 'Wine', 'Breast Cancer']
p_values = [0.186, 0.683, 0.640]
colors = ['#3498db', '#3498db', '#27ae60']

fig, ax = plt.subplots(figsize=(10, 6))

bars = ax.barh(datasets, p_values, color=colors, 
               edgecolor='black', linewidth=1.5, alpha=0.85)

# Significance threshold line
ax.axvline(0.05, color='red', linestyle='--', linewidth=2, 
           label='α = 0.05 (significance threshold)')

# Add p-value labels
for i, (dataset, p) in enumerate(zip(datasets, p_values)):
    ax.text(p + 0.03, i, f'p = {p:.3f}', va='center', 
            fontsize=11, fontweight='bold')

ax.set_xlabel('p-value (Paired t-test, 20-fold CV)', 
              fontsize=12, fontweight='bold')
ax.set_title('Statistical Equivalence to CART\n(All p > 0.05 = No Significant Difference)', 
             fontsize=13, fontweight='bold', pad=15)
ax.legend(loc='lower right', fontsize=10)
ax.set_xlim(0, 0.75)
ax.grid(axis='x', alpha=0.3, linestyle='--')

plt.tight_layout()
plt.savefig('results/figures/statistical_significance.png', dpi=300)
```

## Pareto Front Visualization

### 1. Accuracy vs Interpretability Trade-off

Show Pareto-optimal solutions:

```python
# Results from Pareto optimization
results = [
    {'accuracy': 0.91, 'interpretability': 0.85, 'nodes': 5},
    {'accuracy': 0.93, 'interpretability': 0.72, 'nodes': 9},
    {'accuracy': 0.94, 'interpretability': 0.65, 'nodes': 12},
    {'accuracy': 0.95, 'interpretability': 0.55, 'nodes': 18},
]

fig, ax = plt.subplots(figsize=(10, 8))

# Extract data
accuracies = [r['accuracy'] for r in results]
interps = [r['interpretability'] for r in results]
nodes = [r['nodes'] for r in results]

# Scatter plot with node size as color
scatter = ax.scatter(interps, accuracies, 
                    s=300, c=nodes, cmap='viridis',
                    alpha=0.7, edgecolors='black', linewidth=2)

# Connect Pareto front
ax.plot(interps, accuracies, 'k--', alpha=0.3, linewidth=1.5)

# Annotate points
for i, r in enumerate(results):
    ax.annotate(f"{r['nodes']} nodes", 
                xy=(r['interpretability'], r['accuracy']),
                xytext=(10, 10), textcoords='offset points',
                fontsize=9, bbox=dict(boxstyle='round,pad=0.3', 
                                     facecolor='yellow', alpha=0.7))

ax.set_xlabel('Interpretability Score', fontsize=13, fontweight='bold')
ax.set_ylabel('Test Accuracy', fontsize=13, fontweight='bold')
ax.set_title('Pareto Front: Accuracy vs Interpretability Trade-off', 
             fontsize=14, fontweight='bold', pad=15)
ax.grid(True, alpha=0.3, linestyle='--')

# Colorbar
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('Number of Nodes', fontsize=11)

# Highlight ideal region
ax.axhspan(0.90, 0.95, alpha=0.05, color='green', label='High Accuracy Zone')
ax.axvspan(0.70, 0.90, alpha=0.05, color='blue', label='High Interpretability Zone')

plt.tight_layout()
plt.savefig('results/figures/pareto_front.png', dpi=300)
```

### 2. 3D Pareto Front

Multiple objectives visualization:

```python
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot(111, projection='3d')

# Extract data
accuracies = [r['accuracy'] for r in results]
interps = [r['interpretability'] for r in results]
depths = [r.get('depth', 3) for r in results]

scatter = ax.scatter(accuracies, interps, depths,
                    s=200, c=nodes, cmap='plasma',
                    alpha=0.7, edgecolors='black', linewidth=1.5)

ax.set_xlabel('Accuracy', fontsize=11, fontweight='bold')
ax.set_ylabel('Interpretability', fontsize=11, fontweight='bold')
ax.set_zlabel('Tree Depth', fontsize=11, fontweight='bold')
ax.set_title('3D Pareto Front', fontsize=13, fontweight='bold', pad=20)

plt.colorbar(scatter, ax=ax, label='Nodes', shrink=0.5)
plt.savefig('results/figures/pareto_3d.png', dpi=300)
```

## Feature Importance Visualization

### 1. Feature Usage Frequency

```python
from ga_trees.evaluation.feature_importance import FeatureImportanceAnalyzer

# Calculate importance
analyzer = FeatureImportanceAnalyzer()
importance = analyzer.calculate_feature_depth_importance(best_tree)

# Sort by importance
features = list(importance.keys())
scores = list(importance.values())
sorted_pairs = sorted(zip(scores, features), reverse=True)
scores, features = zip(*sorted_pairs)

# Get feature names
if feature_names:
    labels = [feature_names[f] for f in features]
else:
    labels = [f"Feature {f}" for f in features]

# Plot
fig, ax = plt.subplots(figsize=(10, 6))

bars = ax.barh(range(len(labels)), scores, color='skyblue', 
               edgecolor='black', linewidth=1.2)

ax.set_yticks(range(len(labels)))
ax.set_yticklabels(labels)
ax.set_xlabel('Importance Score', fontsize=12, fontweight='bold')
ax.set_title('Feature Importance in GA Tree', fontsize=14, fontweight='bold')
ax.grid(axis='x', alpha=0.3, linestyle='--')

# Color top 3 features
for i in range(min(3, len(bars))):
    bars[i].set_color('#2ecc71')
    bars[i].set_alpha(0.9)

plt.tight_layout()
plt.savefig('results/figures/feature_importance.png', dpi=300)
```

### 2. Feature Depth Distribution

```python
from collections import defaultdict

# Collect feature usage by depth
feature_depths = defaultdict(list)

def collect_feature_depths(node):
    if node.is_internal():
        feature_depths[node.feature_idx].append(node.depth)
        if node.left_child:
            collect_feature_depths(node.left_child)
        if node.right_child:
            collect_feature_depths(node.right_child)

collect_feature_depths(best_tree.root)

# Plot
fig, ax = plt.subplots(figsize=(10, 6))

for feat_idx, depths in feature_depths.items():
    label = feature_names[feat_idx] if feature_names else f"Feature {feat_idx}"
    ax.scatter([feat_idx] * len(depths), depths, s=100, alpha=0.6, label=label)

ax.set_xlabel('Feature Index', fontsize=12, fontweight='bold')
ax.set_ylabel('Depth in Tree', fontsize=12, fontweight='bold')
ax.set_title('Feature Usage by Depth', fontsize=14, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.invert_yaxis()  # Root at top

plt.tight_layout()
plt.savefig('results/figures/feature_depth_distribution.png', dpi=300)
```

## Confusion Matrix Visualization

```python
from sklearn.metrics import confusion_matrix

# Predictions
y_pred = predictor.predict(best_tree, X_test)

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Plot
fig, ax = plt.subplots(figsize=(8, 6))

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_names, yticklabels=class_names,
            cbar_kws={'label': 'Count'}, ax=ax)

ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold', pad=15)

plt.tight_layout()
plt.savefig('results/figures/confusion_matrix.png', dpi=300)
```

## Publication-Quality Figures

### Using the Built-in Visualizer

```bash
# Generate all publication figures
python scripts/visualize_comprehensive.py
```

**Output:**

- `paper_fig1_size_reduction.png` - Tree size comparison
- `paper_fig2_statistical_equiv.png` - P-value plot
- `paper_fig3_pareto_tradeoff.png` - Accuracy-interpretability scatter
- `paper_table_summary.png` - Results table

### Custom Publication Style

```python
# Set publication style
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 13,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight'
})

sns.set_palette("colorblind")

# Now create your plots...
```

## Interactive Visualizations

### Plotly for Interactive Plots

```python
import plotly.graph_objects as go
import plotly.express as px

# Interactive evolution plot
fig = go.Figure()

fig.add_trace(go.Scatter(
    x=list(range(len(history['best_fitness']))),
    y=history['best_fitness'],
    mode='lines+markers',
    name='Best Fitness',
    line=dict(color='#2ecc71', width=3),
    marker=dict(size=6)
))

fig.add_trace(go.Scatter(
    x=list(range(len(history['avg_fitness']))),
    y=history['avg_fitness'],
    mode='lines',
    name='Average Fitness',
    line=dict(color='#3498db', width=2),
    fill='tonexty'
))

fig.update_layout(
    title='GA Evolution: Interactive Fitness Plot',
    xaxis_title='Generation',
    yaxis_title='Fitness',
    hovermode='x unified',
    template='plotly_white'
)

fig.write_html('results/figures/fitness_evolution_interactive.html')
fig.show()
```

### Jupyter Notebook Widgets

```python
from ipywidgets import interact, IntSlider

@interact(generation=IntSlider(min=0, max=len(history['best_fitness'])-1, value=0))
def plot_generation(generation):
    """Interactive generation explorer."""
    fitness = history['best_fitness'][generation]
    avg_fitness = history['avg_fitness'][generation]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(['Best', 'Average'], [fitness, avg_fitness], color=['#2ecc71', '#3498db'])
    ax.set_title(f'Generation {generation}: Fitness Comparison')
    ax.set_ylabel('Fitness')
    plt.show()
```

## Saving All Visualizations

Automate figure generation:

```python
def generate_all_visualizations(best_tree, history, results, output_dir='results/figures'):
    """Generate complete visualization suite."""
    from pathlib import Path
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Generating visualizations...")
    
    # 1. Tree structure
    TreeVisualizer.visualize_tree(best_tree, save_path=str(output_dir / 'tree_structure'))
    print("✓ Tree structure")
    
    # 2. Evolution history
    plot_fitness_evolution(history, save_path=str(output_dir / 'fitness_evolution.png'))
    print("✓ Fitness evolution")
    
    # 3. Results comparison
    plot_results_comparison(results, save_path=str(output_dir / 'results_comparison.png'))
    print("✓ Results comparison")
    
    # 4. Feature importance
    plot_feature_importance(best_tree, save_path=str(output_dir / 'feature_importance.png'))
    print("✓ Feature importance")
    
    # 5. Confusion matrix
    plot_confusion_matrix(y_test, y_pred, save_path=str(output_dir / 'confusion_matrix.png'))
    print("✓ Confusion matrix")
    
    print(f"\n✓ All visualizations saved to: {output_dir}")

# Use it
generate_all_visualizations(best_tree, history, results)
```

## Next Steps

- **Analyze Results**: See Statistical Tests
- **Custom Visualizations**: Create custom plots
- **Export for Papers**: Format for publications
- **Interactive Dashboards**: Build web interfaces