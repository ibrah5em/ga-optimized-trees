"""
Test the optimized hyperparameters from Optuna.

This script trains with the best hyperparameters and compares to baseline.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import numpy as np
import yaml
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.tree import DecisionTreeClassifier

from ga_trees.ga.engine import GAEngine, GAConfig, TreeInitializer, Mutation
from ga_trees.fitness.calculator import FitnessCalculator, TreePredictor


def load_optimized_config():
    """Load optimized config from Optuna."""
    with open('configs/optimized.yaml', 'r') as f:
        config = yaml.safe_load(f)
    return config


def run_with_optimized_params(X, y, n_folds=5):
    """Run GA with optimized hyperparameters."""
    print("\n" + "="*70)
    print("TESTING OPTIMIZED HYPERPARAMETERS")
    print("="*70)
    
    # Load config
    config = load_optimized_config()
    
    print("\nOptimized Configuration:")
    print(yaml.dump(config, default_flow_style=False))
    
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    results = {
        'test_acc': [], 'test_f1': [],
        'nodes': [], 'depth': [],
        'train_acc': [], 'time': []
    }
    
    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
        print(f"\nFold {fold}/{n_folds}:")
        
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Standardize
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        # Setup with optimized params
        n_features = X_train.shape[1]
        feature_ranges = {i: (X_train[:, i].min(), X_train[:, i].max()) 
                         for i in range(n_features)}
        
        ga_config = GAConfig(**config['ga'])
        initializer = TreeInitializer(n_features=n_features, n_classes=2, **config['tree'])
        fitness_calc = FitnessCalculator(**config['fitness'])
        mutation = Mutation(n_features=n_features, feature_ranges=feature_ranges)
        
        # Train
        import time
        start = time.time()
        ga_engine = GAEngine(ga_config, initializer, 
                           fitness_calc.calculate_fitness, mutation)
        best_tree = ga_engine.evolve(X_train, y_train, verbose=False)
        elapsed = time.time() - start
        
        # Evaluate
        predictor = TreePredictor()
        y_train_pred = predictor.predict(best_tree, X_train)
        y_test_pred = predictor.predict(best_tree, X_test)
        
        train_acc = accuracy_score(y_train, y_train_pred)
        test_acc = accuracy_score(y_test, y_test_pred)
        test_f1 = f1_score(y_test, y_test_pred, average='weighted')
        
        results['train_acc'].append(train_acc)
        results['test_acc'].append(test_acc)
        results['test_f1'].append(test_f1)
        results['nodes'].append(best_tree.get_num_nodes())
        results['depth'].append(best_tree.get_depth())
        results['time'].append(elapsed)
        
        print(f"  Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}")
        print(f"  Nodes: {best_tree.get_num_nodes()}, Depth: {best_tree.get_depth()}")
        print(f"  Time: {elapsed:.1f}s")
    
    return results


def run_baseline(X, y, n_folds=5):
    """Run CART baseline."""
    print("\n" + "="*70)
    print("CART BASELINE")
    print("="*70)
    
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    results = {'test_acc': [], 'test_f1': [], 'nodes': [], 'depth': []}
    
    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        cart = DecisionTreeClassifier(max_depth=5, random_state=42)
        cart.fit(X_train, y_train)
        y_pred = cart.predict(X_test)
        
        results['test_acc'].append(accuracy_score(y_test, y_pred))
        results['test_f1'].append(f1_score(y_test, y_pred, average='weighted'))
        results['nodes'].append(cart.tree_.node_count)
        results['depth'].append(cart.tree_.max_depth)
    
    return results


def main():
    """Run comparison with optimized hyperparameters."""
    print("\n" + "🚀 "*35)
    print("TESTING OPTIMIZED HYPERPARAMETERS ON BREAST CANCER")
    print("🚀 "*35)
    
    # Load data
    X, y = load_breast_cancer(return_X_y=True)
    print(f"\nDataset: {X.shape[0]} samples, {X.shape[1]} features")
    
    # Run with optimized params
    ga_results = run_with_optimized_params(X, y, n_folds=5)
    
    # Run baseline
    cart_results = run_baseline(X, y, n_folds=5)
    
    # Compare
    print("\n" + "="*70)
    print("COMPARISON: OPTIMIZED GA vs CART")
    print("="*70)
    
    import pandas as pd
    
    comparison = pd.DataFrame({
        'Model': ['Optimized GA', 'CART'],
        'Test Accuracy': [
            f"{np.mean(ga_results['test_acc']):.4f} ± {np.std(ga_results['test_acc']):.4f}",
            f"{np.mean(cart_results['test_acc']):.4f} ± {np.std(cart_results['test_acc']):.4f}"
        ],
        'Test F1': [
            f"{np.mean(ga_results['test_f1']):.4f} ± {np.std(ga_results['test_f1']):.4f}",
            f"{np.mean(cart_results['test_f1']):.4f} ± {np.std(cart_results['test_f1']):.4f}"
        ],
        'Nodes': [
            f"{np.mean(ga_results['nodes']):.1f} ± {np.std(ga_results['nodes']):.1f}",
            f"{np.mean(cart_results['nodes']):.1f} ± {np.std(cart_results['nodes']):.1f}"
        ],
        'Depth': [
            f"{np.mean(ga_results['depth']):.1f} ± {np.std(ga_results['depth']):.1f}",
            f"{np.mean(cart_results['depth']):.1f} ± {np.std(cart_results['depth']):.1f}"
        ]
    })
    
    print("\n" + comparison.to_string(index=False))
    
    # Statistical test
    from scipy import stats
    
    ga_acc = ga_results['test_acc']
    cart_acc = cart_results['test_acc']
    
    t_stat, p_value = stats.ttest_rel(ga_acc, cart_acc)
    
    print("\n" + "="*70)
    print("STATISTICAL SIGNIFICANCE")
    print("="*70)
    
    print(f"\nPaired t-test:")
    print(f"  t-statistic: {t_stat:.4f}")
    print(f"  p-value: {p_value:.4f}")
    
    if p_value < 0.05:
        winner = "GA" if np.mean(ga_acc) > np.mean(cart_acc) else "CART"
        print(f"  Result: {winner} is significantly better! ✓")
    else:
        print(f"  Result: No significant difference (both perform similarly)")
    
    # Effect size
    pooled_std = np.sqrt((np.var(ga_acc) + np.var(cart_acc)) / 2)
    cohens_d = (np.mean(ga_acc) - np.mean(cart_acc)) / pooled_std
    
    print(f"\nCohen's d: {cohens_d:.4f}")
    if abs(cohens_d) < 0.2:
        print("  Effect size: Negligible")
    elif abs(cohens_d) < 0.5:
        print("  Effect size: Small")
    elif abs(cohens_d) < 0.8:
        print("  Effect size: Medium")
    else:
        print("  Effect size: Large")
    
    # Improvement analysis
    print("\n" + "="*70)
    print("IMPROVEMENT ANALYSIS")
    print("="*70)
    
    acc_improvement = (np.mean(ga_acc) - np.mean(cart_acc)) * 100
    node_reduction = (1 - np.mean(ga_results['nodes']) / np.mean(cart_results['nodes'])) * 100
    
    print(f"\nAccuracy change: {acc_improvement:+.2f}%")
    if acc_improvement > 0:
        print(f"  → GA is {acc_improvement:.2f}% MORE accurate! 🎉")
    elif acc_improvement > -1:
        print(f"  → GA matches CART (within 1%) ✓")
    else:
        print(f"  → GA trades {abs(acc_improvement):.2f}% accuracy for interpretability")
    
    print(f"\nTree size change: {node_reduction:+.2f}%")
    if node_reduction > 0:
        print(f"  → GA produces {node_reduction:.1f}% SMALLER trees! 🏆")
    else:
        print(f"  → GA produces {abs(node_reduction):.1f}% LARGER trees")
    
    # Overall verdict
    print("\n" + "="*70)
    print("VERDICT")
    print("="*70)
    
    if acc_improvement >= -1 and node_reduction > 10:
        print("\n✓✓ EXCELLENT! GA achieves similar accuracy with smaller trees!")
        print("   This is the ideal outcome for interpretable ML.")
    elif acc_improvement > 1:
        print("\n✓✓ OUTSTANDING! GA beats CART on accuracy!")
        print("   This is better than expected.")
    elif acc_improvement >= -2:
        print("\n✓ GOOD! GA performs competitively.")
        print("  Small accuracy loss is acceptable for interpretability.")
    else:
        print("\n⚠ NEEDS TUNING: Consider increasing accuracy weight.")
    
    # Training time comparison
    if 'time' in ga_results:
        avg_ga_time = np.mean(ga_results['time'])
        print(f"\n⏱ Average training time: {avg_ga_time:.1f}s per fold")
        print(f"  ({avg_ga_time * 5:.0f}s for 5-fold CV)")
    
    print("\n" + "="*70)
    print("Test Complete! 🎉")
    print("="*70)
    
    # Save results
    results_summary = {
        'optimized_ga': {
            'accuracy_mean': np.mean(ga_acc),
            'accuracy_std': np.std(ga_acc),
            'nodes_mean': np.mean(ga_results['nodes']),
            'nodes_std': np.std(ga_results['nodes'])
        },
        'cart': {
            'accuracy_mean': np.mean(cart_acc),
            'accuracy_std': np.std(cart_acc),
            'nodes_mean': np.mean(cart_results['nodes']),
            'nodes_std': np.std(cart_results['nodes'])
        },
        'statistics': {
            't_statistic': float(t_stat),
            'p_value': float(p_value),
            'cohens_d': float(cohens_d)
        }
    }
    
    import json
    with open('results/optimized_comparison.json', 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    print("\n✓ Results saved to: results/optimized_comparison.json")


if __name__ == '__main__':
    main()