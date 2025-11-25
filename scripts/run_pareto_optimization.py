"""Run Pareto optimization and visualize results - FIXED TO USE CONFIG."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import argparse
import yaml
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer, load_iris, load_wine
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

from ga_trees.ga.engine import GAEngine, GAConfig, TreeInitializer, Mutation
from ga_trees.fitness.calculator import FitnessCalculator, TreePredictor


def load_config(config_path='configs/custom.yaml'):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_dataset(name):
    """Load dataset by name."""
    datasets = {
        'iris': load_iris,
        'wine': load_wine,
        'breast_cancer': load_breast_cancer
    }
    return datasets[name](return_X_y=True)


def run_multiple_configs(X_train, y_train, X_test, y_test, base_config, n_configs=10):
    """
    Run GA with different accuracy/interpretability weight combinations
    to approximate Pareto front.
    
    Uses your base config for all GA parameters, only varies fitness weights.
    """
    print("="*70)
    print("PARETO FRONT APPROXIMATION")
    print(f"Base config: {base_config.get('meta', {}).get('description', 'custom')}")
    print("Running GA with different objective weights...")
    print("="*70)
    
    # Generate weight combinations around your base weights
    base_acc_weight = base_config['fitness']['weights']['accuracy']
    
    # Create range: base_weight ± 0.25
    min_weight = max(0.3, base_acc_weight - 0.25)
    max_weight = min(0.95, base_acc_weight + 0.25)
    
    weight_configs = []
    for acc_weight in np.linspace(min_weight, max_weight, n_configs):
        weight_configs.append({
            'accuracy': acc_weight,
            'interpretability': 1.0 - acc_weight
        })
    
    results = []
    
    for i, weights in enumerate(weight_configs, 1):
        print(f"\nConfig {i}/{n_configs}: Accuracy={weights['accuracy']:.2f}, "
              f"Interpretability={weights['interpretability']:.2f}")
        
        # Setup with YOUR config parameters
        n_features = X_train.shape[1]
        n_classes = len(np.unique(y_train))
        feature_ranges = {j: (X_train[:, j].min(), X_train[:, j].max()) 
                         for j in range(n_features)}
        
        # Use your GA config
        ga_config = GAConfig(
            population_size=base_config['ga']['population_size'],
            n_generations=base_config['ga']['n_generations'],
            crossover_prob=base_config['ga']['crossover_prob'],
            mutation_prob=base_config['ga']['mutation_prob'],
            tournament_size=base_config['ga']['tournament_size'],
            elitism_ratio=base_config['ga']['elitism_ratio'],
            mutation_types=base_config['ga']['mutation_types']
        )
        
        # Use your tree constraints
        tree_config = base_config['tree']
        initializer = TreeInitializer(
            n_features=n_features,
            n_classes=n_classes,
            max_depth=tree_config['max_depth'],
            min_samples_split=tree_config['min_samples_split'],
            min_samples_leaf=tree_config['min_samples_leaf']
        )
        
        # Use your interpretability sub-weights, vary main weights only
        fitness_calc = FitnessCalculator(
            mode='weighted_sum',
            accuracy_weight=weights['accuracy'],
            interpretability_weight=weights['interpretability'],
            interpretability_weights=base_config['fitness']['interpretability_weights']
        )
        
        mutation = Mutation(n_features=n_features, feature_ranges=feature_ranges)
        
        # Train
        ga_engine = GAEngine(ga_config, initializer, 
                           fitness_calc.calculate_fitness, mutation)
        best_tree = ga_engine.evolve(X_train, y_train, verbose=False)
        
        # Evaluate
        predictor = TreePredictor()
        y_pred = predictor.predict(best_tree, X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        results.append({
            'accuracy_weight': weights['accuracy'],
            'test_accuracy': accuracy,
            'interpretability': best_tree.interpretability_,
            'nodes': best_tree.get_num_nodes(),
            'depth': best_tree.get_depth(),
            'tree': best_tree
        })
        
        print(f"  → Test Accuracy: {accuracy:.4f}, "
              f"Interpretability: {best_tree.interpretability_:.4f}, "
              f"Nodes: {best_tree.get_num_nodes()}")
    
    return results


def plot_pareto_front(results, dataset_name, base_config, 
                      save_path='results/figures/pareto_front.png'):
    """Visualize approximate Pareto front."""
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Accuracy vs Interpretability
    ax1 = axes[0]
    
    accuracies = [r['test_accuracy'] for r in results]
    interpretabilities = [r['interpretability'] for r in results]
    nodes = [r['nodes'] for r in results]
    
    scatter = ax1.scatter(interpretabilities, accuracies, 
                         s=200, c=nodes, cmap='viridis', 
                         alpha=0.7, edgecolors='black', linewidth=2)
    
    # Highlight your base config point
    base_acc = base_config['fitness']['weights']['accuracy']
    base_result = min(results, key=lambda r: abs(r['accuracy_weight'] - base_acc))
    ax1.scatter(base_result['interpretability'], base_result['test_accuracy'],
               s=400, color='red', marker='*', edgecolors='black', linewidth=2,
               label='Your Config', zorder=10)
    
    ax1.set_xlabel('Interpretability Score', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Test Accuracy', fontsize=13, fontweight='bold')
    ax1.set_title(f'Pareto Front: Accuracy vs Interpretability\n{dataset_name} Dataset', 
                 fontsize=15, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=11)
    
    cbar = plt.colorbar(scatter, ax=ax1)
    cbar.set_label('Number of Nodes', fontsize=11)
    
    # Add annotations for extremes
    min_nodes_idx = np.argmin(nodes)
    max_acc_idx = np.argmax(accuracies)
    
    ax1.annotate('Most Interpretable', 
                xy=(interpretabilities[min_nodes_idx], accuracies[min_nodes_idx]),
                xytext=(20, -20), textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.3'))
    
    ax1.annotate('Most Accurate', 
                xy=(interpretabilities[max_acc_idx], accuracies[max_acc_idx]),
                xytext=(-20, 20), textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.7),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.3'))
    
    # Plot 2: Accuracy vs Tree Size
    ax2 = axes[1]
    
    scatter2 = ax2.scatter(nodes, accuracies, 
                          s=200, c=interpretabilities, cmap='coolwarm',
                          alpha=0.7, edgecolors='black', linewidth=2)
    
    # Highlight your base config
    ax2.scatter(base_result['nodes'], base_result['test_accuracy'],
               s=400, color='red', marker='*', edgecolors='black', linewidth=2,
               label='Your Config', zorder=10)
    
    ax2.set_xlabel('Number of Nodes', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Test Accuracy', fontsize=13, fontweight='bold')
    ax2.set_title(f'Trade-off: Accuracy vs Tree Complexity\n{dataset_name} Dataset', 
                 fontsize=15, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=11)
    
    cbar2 = plt.colorbar(scatter2, ax=ax2)
    cbar2.set_label('Interpretability Score', fontsize=11)
    
    # Add "sweet spot" region
    if accuracies:
        high_acc_threshold = max(accuracies) - 0.02
        ax2.axhspan(high_acc_threshold, max(accuracies) + 0.01, 
                   alpha=0.1, color='green', label='High Accuracy Zone')
    
    if nodes:
        low_complexity = min(nodes) + (max(nodes) - min(nodes)) * 0.3
        ax2.axvspan(min(nodes), low_complexity, 
                   alpha=0.1, color='blue', label='Low Complexity Zone')
    
    plt.tight_layout()
    
    # Save
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Pareto front saved to: {save_path}")
    
    plt.show()


def print_pareto_summary(results, base_config):
    """Print summary of Pareto front."""
    print("\n" + "="*70)
    print("PARETO FRONT SUMMARY")
    print("="*70)
    
    print(f"\n{'Config':<8} {'Acc Weight':<12} {'Test Acc':<12} {'Interp':<12} {'Nodes':<8} {'Depth'}")
    print("-"*70)
    
    base_acc_weight = base_config['fitness']['weights']['accuracy']
    
    for i, r in enumerate(results, 1):
        marker = ' ★' if abs(r['accuracy_weight'] - base_acc_weight) < 0.05 else ''
        print(f"{i:<8} {r['accuracy_weight']:<12.2f} "
              f"{r['test_accuracy']:<12.4f} {r['interpretability']:<12.4f} "
              f"{r['nodes']:<8} {r['depth']}{marker}")
    
    # Find interesting solutions
    print("\n" + "="*70)
    print("INTERESTING SOLUTIONS")
    print("="*70)
    
    best_acc_idx = np.argmax([r['test_accuracy'] for r in results])
    smallest_tree_idx = np.argmin([r['nodes'] for r in results])
    best_interp_idx = np.argmax([r['interpretability'] for r in results])
    
    # Find your config solution
    your_config_idx = min(range(len(results)), 
                         key=lambda i: abs(results[i]['accuracy_weight'] - base_acc_weight))
    
    print(f"\n1. MOST ACCURATE:")
    print(f"   Accuracy: {results[best_acc_idx]['test_accuracy']:.4f}")
    print(f"   Nodes: {results[best_acc_idx]['nodes']}")
    print(f"   Interpretability: {results[best_acc_idx]['interpretability']:.4f}")
    
    print(f"\n2. MOST INTERPRETABLE:")
    print(f"   Accuracy: {results[best_interp_idx]['test_accuracy']:.4f}")
    print(f"   Nodes: {results[best_interp_idx]['nodes']}")
    print(f"   Interpretability: {results[best_interp_idx]['interpretability']:.4f}")
    
    print(f"\n3. SMALLEST TREE:")
    print(f"   Accuracy: {results[smallest_tree_idx]['test_accuracy']:.4f}")
    print(f"   Nodes: {results[smallest_tree_idx]['nodes']}")
    print(f"   Interpretability: {results[smallest_tree_idx]['interpretability']:.4f}")
    
    print(f"\n4. YOUR CONFIG (acc_weight={base_acc_weight:.2f}):")
    print(f"   Accuracy: {results[your_config_idx]['test_accuracy']:.4f}")
    print(f"   Nodes: {results[your_config_idx]['nodes']}")
    print(f"   Interpretability: {results[your_config_idx]['interpretability']:.4f}")


def main():
    """Run Pareto front approximation."""
    parser = argparse.ArgumentParser(description='Run Pareto front analysis')
    parser.add_argument('--config', type=str, default='configs/custom.yaml',
                       help='Path to configuration file')
    parser.add_argument('--dataset', type=str, default='breast_cancer',
                       choices=['iris', 'wine', 'breast_cancer'],
                       help='Dataset to use')
    parser.add_argument('--n-configs', type=int, default=10,
                       help='Number of weight configurations to try')
    args = parser.parse_args()
    
    # Load your config
    config = load_config(args.config)
    
    print("\n" + "="*70)
    print(f"PARETO FRONT ANALYSIS FOR {args.dataset.upper()}")
    print(f"Using config: {args.config}")
    print("="*70)
    
    # Load data
    print(f"\nLoading {args.dataset} dataset...")
    X, y = load_dataset(args.dataset)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Standardize
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    print(f"Train: {X_train.shape[0]} samples")
    print(f"Test: {X_test.shape[0]} samples")
    print(f"Features: {X_train.shape[1]}")
    
    # Run with multiple weight configurations
    results = run_multiple_configs(X_train, y_train, X_test, y_test, 
                                   config, n_configs=args.n_configs)
    
    # Visualize Pareto front
    plot_pareto_front(results, args.dataset, config)
    
    # Print summary
    print_pareto_summary(results, config)
    
    print("\n" + "="*70)
    print("Analysis Complete! 🎉")
    print("="*70)
    print("\nKey Insights:")
    print("• The Pareto front shows optimal trade-offs")
    print("• Your config (★) represents one point on this front")
    print("• No single solution dominates on both objectives")
    print("• Choose solution based on your priorities:")
    print("  - High accuracy needed? → Pick rightmost point")
    print("  - Interpretability critical? → Pick topmost point")
    print(f"  - Your current choice → {config['fitness']['weights']['accuracy']:.0%} accuracy weight")


if __name__ == '__main__':
    main()