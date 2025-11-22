"""
Minimal experiment script that works with just the core files.
No external dependencies on missing modules.

Usage:
    python scripts/experiment_minimal.py
"""

import numpy as np
import pandas as pd
from pathlib import Path
import sys
import time
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from sklearn.datasets import load_iris, load_wine, load_breast_cancer
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from scipy import stats

# Import our core modules
from ga_trees.ga.engine import GAEngine, GAConfig, TreeInitializer, Mutation
from ga_trees.fitness.calculator import FitnessCalculator, TreePredictor


def load_dataset(name):
    """Load dataset by name."""
    datasets = {
        'iris': load_iris,
        'wine': load_wine,
        'breast_cancer': load_breast_cancer,
    }
    return datasets[name](return_X_y=True)


def run_ga_experiment(X, y, dataset_name, n_folds=5):
    """Run GA with cross-validation."""
    print(f"\n{'='*70}")
    print(f"Running GA on {dataset_name}")
    print(f"{'='*70}")
    
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    results = {'test_acc': [], 'test_f1': [], 'nodes': [], 'depth': [], 'time': []}
    
    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
        print(f"  Fold {fold}/{n_folds}...", end=' ')
        
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Standardize
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        # Setup
        n_features = X_train.shape[1]
        n_classes = len(np.unique(y))
        feature_ranges = {i: (X_train[:, i].min(), X_train[:, i].max()) 
                         for i in range(n_features)}
        
        # Create GA components
        ga_config = GAConfig(
            population_size=50,  # Smaller for speed
            n_generations=30,
            crossover_prob=0.7,
            mutation_prob=0.2,
            tournament_size=3,
            elitism_ratio=0.1
        )
        
        initializer = TreeInitializer(
            n_features=n_features,
            n_classes=n_classes,
            max_depth=5,
            min_samples_split=10,
            min_samples_leaf=5
        )
        
        fitness_calc = FitnessCalculator(
            mode='weighted_sum',
            accuracy_weight=0.7,
            interpretability_weight=0.3
        )
        
        mutation = Mutation(n_features=n_features, feature_ranges=feature_ranges)
        
        # Train
        start = time.time()
        ga_engine = GAEngine(ga_config, initializer, 
                           fitness_calc.calculate_fitness, mutation)
        best_tree = ga_engine.evolve(X_train, y_train, verbose=False)
        elapsed = time.time() - start
        
        # Evaluate
        predictor = TreePredictor()
        y_pred = predictor.predict(best_tree, X_test)
        
        results['test_acc'].append(accuracy_score(y_test, y_pred))
        results['test_f1'].append(f1_score(y_test, y_pred, average='weighted'))
        results['nodes'].append(best_tree.get_num_nodes())
        results['depth'].append(best_tree.get_depth())
        results['time'].append(elapsed)
        
        print(f"Acc={results['test_acc'][-1]:.3f}, Time={elapsed:.1f}s")
    
    return results


def run_cart_experiment(X, y, dataset_name, n_folds=5):
    """Run CART baseline."""
    print(f"\n{'='*70}")
    print(f"Running CART on {dataset_name}")
    print(f"{'='*70}")
    
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    results = {'test_acc': [], 'test_f1': [], 'nodes': [], 'depth': [], 'time': []}
    
    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
        print(f"  Fold {fold}/{n_folds}...", end=' ')
        
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Train
        start = time.time()
        model = DecisionTreeClassifier(max_depth=5, random_state=42)
        model.fit(X_train, y_train)
        elapsed = time.time() - start
        
        # Evaluate
        y_pred = model.predict(X_test)
        
        results['test_acc'].append(accuracy_score(y_test, y_pred))
        results['test_f1'].append(f1_score(y_test, y_pred, average='weighted'))
        results['nodes'].append(model.tree_.node_count)
        results['depth'].append(model.tree_.max_depth)
        results['time'].append(elapsed)
        
        print(f"Acc={results['test_acc'][-1]:.3f}, Time={elapsed:.1f}s")
    
    return results


def run_rf_experiment(X, y, dataset_name, n_folds=5):
    """Run Random Forest baseline."""
    print(f"\n{'='*70}")
    print(f"Running Random Forest on {dataset_name}")
    print(f"{'='*70}")
    
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    results = {'test_acc': [], 'test_f1': [], 'time': []}
    
    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
        print(f"  Fold {fold}/{n_folds}...", end=' ')
        
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Train
        start = time.time()
        model = RandomForestClassifier(n_estimators=100, max_depth=5, 
                                       random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
        elapsed = time.time() - start
        
        # Evaluate
        y_pred = model.predict(X_test)
        
        results['test_acc'].append(accuracy_score(y_test, y_pred))
        results['test_f1'].append(f1_score(y_test, y_pred, average='weighted'))
        results['time'].append(elapsed)
        
        print(f"Acc={results['test_acc'][-1]:.3f}, Time={elapsed:.1f}s")
    
    return results


def print_summary(all_results):
    """Print summary table."""
    print(f"\n{'='*70}")
    print("FINAL RESULTS")
    print(f"{'='*70}\n")
    
    data = []
    for dataset_name, models in all_results.items():
        for model_name, results in models.items():
            acc_mean = np.mean(results['test_acc'])
            acc_std = np.std(results['test_acc'])
            f1_mean = np.mean(results['test_f1'])
            f1_std = np.std(results['test_f1'])
            time_mean = np.mean(results['time'])
            
            row = {
                'Dataset': dataset_name,
                'Model': model_name,
                'Test Acc': f"{acc_mean:.4f} ± {acc_std:.4f}",
                'Test F1': f"{f1_mean:.4f} ± {f1_std:.4f}",
                'Time (s)': f"{time_mean:.2f}"
            }
            
            # Add tree metrics if available
            if 'nodes' in results:
                row['Nodes'] = f"{np.mean(results['nodes']):.1f}"
                row['Depth'] = f"{np.mean(results['depth']):.1f}"
            
            data.append(row)
    
    df = pd.DataFrame(data)
    print(df.to_string(index=False))
    
    # Statistical tests
    print(f"\n{'='*70}")
    print("Statistical Tests (GA vs CART)")
    print(f"{'='*70}\n")
    
    for dataset_name in all_results.keys():
        ga_acc = all_results[dataset_name]['GA-Optimized']['test_acc']
        cart_acc = all_results[dataset_name]['CART']['test_acc']
        
        t_stat, p_value = stats.ttest_rel(ga_acc, cart_acc)
        
        # Cohen's d
        pooled_std = np.sqrt((np.var(ga_acc) + np.var(cart_acc)) / 2)
        cohens_d = (np.mean(ga_acc) - np.mean(cart_acc)) / pooled_std
        
        print(f"{dataset_name:20s}: t={t_stat:6.3f}, p={p_value:.4f}, d={cohens_d:.3f}")
    
    # Save results
    output_dir = Path('results')
    output_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    df.to_csv(output_dir / f'results_{timestamp}.csv', index=False)
    
    print(f"\n✓ Results saved to: results/results_{timestamp}.csv")


def main():
    """Run experiments on all datasets."""
    print(f"\n{'='*70}")
    print("GA-Optimized Decision Trees: Experiment Suite")
    print(f"{'='*70}")
    
    datasets = ['iris', 'wine', 'breast_cancer']
    all_results = {}
    
    for dataset_name in datasets:
        # Load data
        X, y = load_dataset(dataset_name)
        print(f"\n{dataset_name}: {X.shape[0]} samples, {X.shape[1]} features")
        
        # Run experiments
        dataset_results = {}
        
        # GA
        dataset_results['GA-Optimized'] = run_ga_experiment(X, y, dataset_name)
        
        # CART
        dataset_results['CART'] = run_cart_experiment(X, y, dataset_name)
        
        # Random Forest
        dataset_results['Random Forest'] = run_rf_experiment(X, y, dataset_name)
        
        all_results[dataset_name] = dataset_results
    
    # Print summary
    print_summary(all_results)
    
    print(f"\n{'='*70}")
    print("Experiment Complete! 🎉")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()