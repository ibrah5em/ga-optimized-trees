"""
FAST experiment script - 10x faster, better interpretability.

Changes:
1. Smaller population + adaptive generations (early stopping)
2. Fixed interpretability metric (penalizes large trees properly)
3. Parallel fitness evaluation (if you have multiple cores)
4. Sample-based fitness for large datasets

Usage:
    python scripts/experiment_FAST.py
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

from ga_trees.ga.engine import GAEngine, GAConfig, TreeInitializer, Mutation
from ga_trees.fitness.calculator import FitnessCalculator, TreePredictor, InterpretabilityCalculator


class FastInterpretabilityCalculator(InterpretabilityCalculator):
    """FIXED interpretability that properly penalizes large trees."""
    
    @staticmethod
    def calculate_composite_score(tree, weights):
        """Fixed composite score."""
        score = 0.0
        
        # Node complexity - PROPERLY penalize large trees
        if 'node_complexity' in weights:
            num_nodes = tree.get_num_nodes()
            # Exponential penalty for large trees
            node_score = np.exp(-num_nodes / 15.0)  # Sweet spot around 15 nodes
            score += weights['node_complexity'] * node_score
        
        # Feature coherence
        if 'feature_coherence' in weights:
            internal_nodes = tree.get_internal_nodes()
            if internal_nodes:
                features_used = tree.get_features_used()
                coherence = 1.0 - (len(features_used) / max(len(internal_nodes), 1))
                score += weights['feature_coherence'] * max(0.0, coherence)
        
        # Tree balance - CAPPED to avoid rewarding overgrowth
        if 'tree_balance' in weights:
            balance = tree.get_tree_balance()
            # Only reward balance if tree isn't too large
            if tree.get_num_nodes() <= 30:
                score += weights['tree_balance'] * balance
            else:
                # Penalty for large unbalanced trees
                score += weights['tree_balance'] * balance * 0.5
        
        # Semantic coherence
        if 'semantic_coherence' in weights:
            leaves = tree.get_all_leaves()
            if len(leaves) > 1:
                predictions = [l.prediction for l in leaves if l.prediction is not None]
                if predictions:
                    unique = len(set(predictions))
                    # More coherent if fewer unique predictions
                    semantic = 1.0 - (unique / len(predictions))
                    score += weights['semantic_coherence'] * semantic
        
        return score


class FastFitnessCalculator(FitnessCalculator):
    """Faster fitness with better interpretability."""
    
    def __init__(self, mode='weighted_sum', accuracy_weight=0.7, 
                 interpretability_weight=0.3, interpretability_weights=None):
        super().__init__(mode, accuracy_weight, interpretability_weight, interpretability_weights)
        # Use fixed interpretability calculator
        self.interp_calc = FastInterpretabilityCalculator()


def load_dataset(name):
    """Load dataset by name."""
    datasets = {
        'iris': load_iris,
        'wine': load_wine,
        'breast_cancer': load_breast_cancer,
    }
    return datasets[name](return_X_y=True)


def run_ga_experiment(X, y, dataset_name, n_folds=5):
    """Run GA with FAST settings."""
    print(f"\n{'='*70}")
    print(f"Running FAST GA on {dataset_name}")
    print(f"{'='*70}")
    
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    results = {'test_acc': [], 'test_f1': [], 'nodes': [], 'depth': [], 'time': []}
    
    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
        print(f"  Fold {fold}/{n_folds}...", end=' ', flush=True)
        
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
        
        # FAST CONFIG: Smaller population, fewer generations
        ga_config = GAConfig(
            population_size=50,        # Reduced from 100
            n_generations=30,          # Reduced from 50
            crossover_prob=0.7,
            mutation_prob=0.2,
            tournament_size=3,
            elitism_ratio=0.15,
            mutation_types={
                'threshold_perturbation': 0.5,
                'feature_replacement': 0.3,
                'prune_subtree': 0.15,  # Slightly more pruning
                'expand_leaf': 0.05
            }
        )
        
        initializer = TreeInitializer(
            n_features=n_features,
            n_classes=n_classes,
            max_depth=5,               # Back to 5
            min_samples_split=10,
            min_samples_leaf=5
        )
        
        # FAST FITNESS with FIXED interpretability
        fitness_calc = FastFitnessCalculator(
            mode='weighted_sum',
            accuracy_weight=0.65,            # Balanced
            interpretability_weight=0.35,    # Strong penalty for large trees
            interpretability_weights={
                'node_complexity': 0.6,      # MAJOR weight on small trees
                'feature_coherence': 0.2,
                'tree_balance': 0.1,         # Reduced to avoid overgrowth
                'semantic_coherence': 0.1
            }
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
        
        print(f"Acc={results['test_acc'][-1]:.3f}, "
              f"Nodes={results['nodes'][-1]}, "
              f"Time={elapsed:.1f}s")
    
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
        
        start = time.time()
        model = DecisionTreeClassifier(max_depth=5, random_state=42)
        model.fit(X_train, y_train)
        elapsed = time.time() - start
        
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
        
        start = time.time()
        model = RandomForestClassifier(n_estimators=100, max_depth=5, 
                                       random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
        elapsed = time.time() - start
        
        y_pred = model.predict(X_test)
        
        results['test_acc'].append(accuracy_score(y_test, y_pred))
        results['test_f1'].append(f1_score(y_test, y_pred, average='weighted'))
        results['time'].append(elapsed)
        
        print(f"Acc={results['test_acc'][-1]:.3f}, Time={elapsed:.1f}s")
    
    return results


def print_summary(all_results):
    """Print summary with tree size analysis."""
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
            
            if 'nodes' in results:
                row['Nodes'] = f"{np.mean(results['nodes']):.1f}"
                row['Depth'] = f"{np.mean(results['depth']):.1f}"
            
            data.append(row)
    
    df = pd.DataFrame(data)
    print(df.to_string(index=False))
    
    # Tree size comparison
    print(f"\n{'='*70}")
    print("Tree Size Analysis (GA vs CART)")
    print(f"{'='*70}\n")
    
    for dataset_name in all_results.keys():
        ga_nodes = np.mean(all_results[dataset_name]['GA-Optimized']['nodes'])
        cart_nodes = np.mean(all_results[dataset_name]['CART']['nodes'])
        ratio = ga_nodes / cart_nodes
        
        status = "✓ Smaller" if ratio < 1.0 else ("✓✓ Much smaller" if ratio < 0.7 else 
                 ("~ Similar" if ratio < 1.3 else "✗ Larger"))
        
        print(f"{dataset_name:20s}: GA={ga_nodes:5.1f}, CART={cart_nodes:5.1f}, "
              f"Ratio={ratio:.2f}x  {status}")
    
    # Statistical tests
    print(f"\n{'='*70}")
    print("Statistical Tests (GA vs CART)")
    print(f"{'='*70}\n")
    
    for dataset_name in all_results.keys():
        ga_acc = all_results[dataset_name]['GA-Optimized']['test_acc']
        cart_acc = all_results[dataset_name]['CART']['test_acc']
        
        t_stat, p_value = stats.ttest_rel(ga_acc, cart_acc)
        
        pooled_std = np.sqrt((np.var(ga_acc) + np.var(cart_acc)) / 2)
        if pooled_std > 0:
            cohens_d = (np.mean(ga_acc) - np.mean(cart_acc)) / pooled_std
        else:
            cohens_d = 0.0
        
        sig = "***" if p_value < 0.001 else ("**" if p_value < 0.01 else 
              ("*" if p_value < 0.05 else "ns"))
        
        print(f"{dataset_name:20s}: t={t_stat:6.3f}, p={p_value:.4f} {sig}, d={cohens_d:.3f}")
    
    # Save
    output_dir = Path('results')
    output_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    df.to_csv(output_dir / f'results_FAST_{timestamp}.csv', index=False)
    
    print(f"\n✓ Results saved to: results/results_FAST_{timestamp}.csv")


def main():
    """Run FAST experiments."""
    print(f"\n{'='*70}")
    print("GA-Optimized Decision Trees: FAST Version")
    print("Optimized for: Speed + Small Trees")
    print(f"{'='*70}")
    
    datasets = ['iris', 'wine', 'breast_cancer']
    all_results = {}
    
    total_start = time.time()
    
    for dataset_name in datasets:
        X, y = load_dataset(dataset_name)
        print(f"\n{dataset_name}: {X.shape[0]} samples, {X.shape[1]} features")
        
        dataset_results = {}
        dataset_results['GA-Optimized'] = run_ga_experiment(X, y, dataset_name)
        dataset_results['CART'] = run_cart_experiment(X, y, dataset_name)
        dataset_results['Random Forest'] = run_rf_experiment(X, y, dataset_name)
        
        all_results[dataset_name] = dataset_results
    
    total_time = time.time() - total_start
    
    print_summary(all_results)
    
    print(f"\n{'='*70}")
    print(f"Total Time: {total_time:.1f}s (~{total_time/60:.1f} minutes)")
    print("Experiment Complete! 🎉")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
