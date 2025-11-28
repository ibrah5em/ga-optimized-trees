"""
UPDATED experiment.py with enhanced dataset loader support

Now supports 25+ datasets including OpenML datasets!

Usage:
    # Use any sklearn dataset
    python scripts/experiment.py --config configs/custom.yaml --datasets iris wine breast_cancer
    
    # Use OpenML datasets
    python scripts/experiment.py --config configs/custom.yaml --datasets heart titanic sonar
    
    # Mix both
    python scripts/experiment.py --config configs/custom.yaml --datasets iris heart titanic
"""

import numpy as np
import pandas as pd
from pathlib import Path
import sys
import time
import argparse
import yaml
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from scipy import stats

from ga_trees.ga.engine import GAEngine, GAConfig, TreeInitializer, Mutation
from ga_trees.fitness.calculator import FitnessCalculator, TreePredictor, InterpretabilityCalculator


# Import the integration helper
try:
    from dataset_integration import load_any_dataset
except ImportError:
    # Fallback to basic sklearn datasets
    from sklearn.datasets import load_iris, load_wine, load_breast_cancer
    
    def load_any_dataset(name: str, standardize: bool = False):
        """Fallback loader for sklearn datasets only."""
        datasets = {
            'iris': load_iris,
            'wine': load_wine,
            'breast_cancer': load_breast_cancer
        }
        if name not in datasets:
            raise ValueError(f"Dataset '{name}' not available. Install dataset_loader.")
        return datasets[name](return_X_y=True)


class FastInterpretabilityCalculator(InterpretabilityCalculator):
    """FIXED interpretability that properly penalizes large trees."""
    
    @staticmethod
    def calculate_composite_score(tree, weights):
        """Fixed composite score."""
        score = 0.0
        
        # Node complexity - PROPERLY penalize large trees
        if 'node_complexity' in weights:
            num_nodes = tree.get_num_nodes()
            node_score = np.exp(-num_nodes / 15.0)
            score += weights['node_complexity'] * node_score
        
        # Feature coherence
        if 'feature_coherence' in weights:
            internal_nodes = tree.get_internal_nodes()
            if internal_nodes:
                features_used = tree.get_features_used()
                coherence = 1.0 - (len(features_used) / max(len(internal_nodes), 1))
                score += weights['feature_coherence'] * max(0.0, coherence)
        
        # Tree balance
        if 'tree_balance' in weights:
            balance = tree.get_tree_balance()
            if tree.get_num_nodes() <= 30:
                score += weights['tree_balance'] * balance
            else:
                score += weights['tree_balance'] * balance * 0.5
        
        # Semantic coherence
        if 'semantic_coherence' in weights:
            leaves = tree.get_all_leaves()
            if len(leaves) > 1:
                predictions = [l.prediction for l in leaves if l.prediction is not None]
                if predictions:
                    unique = len(set(predictions))
                    semantic = 1.0 - (unique / len(predictions))
                    score += weights['semantic_coherence'] * semantic
        
        return score


class FastFitnessCalculator(FitnessCalculator):
    """Faster fitness with better interpretability."""
    
    def __init__(self, mode='weighted_sum', accuracy_weight=0.7, 
                 interpretability_weight=0.3, interpretability_weights=None):
        super().__init__(mode, accuracy_weight, interpretability_weight, interpretability_weights)
        self.interp_calc = FastInterpretabilityCalculator()


def load_config(config_path=None):
    """Load configuration from YAML file or use defaults."""
    default_config = {
        'ga': {
            'population_size': 50,
            'n_generations': 30,
            'crossover_prob': 0.7,
            'mutation_prob': 0.2,
            'tournament_size': 3,
            'elitism_ratio': 0.15,
            'mutation_types': {
                'threshold_perturbation': 0.5,
                'feature_replacement': 0.3,
                'prune_subtree': 0.15,
                'expand_leaf': 0.05
            }
        },
        'tree': {
            'max_depth': 5,
            'min_samples_split': 10,
            'min_samples_leaf': 5
        },
        'fitness': {
            'mode': 'weighted_sum',
            'accuracy_weight': 0.65,
            'interpretability_weight': 0.35,
            'interpretability_weights': {
                'node_complexity': 0.6,
                'feature_coherence': 0.2,
                'tree_balance': 0.1,
                'semantic_coherence': 0.1
            }
        },
        'experiment': {
            'datasets': ['iris', 'wine', 'breast_cancer'],
            'cv_folds': 5,
            'random_state': 42
        }
    }
    
    if config_path and Path(config_path).exists():
        print(f"Loading configuration from: {config_path}")
        with open(config_path, 'r') as f:
            user_config = yaml.safe_load(f)
        config = _merge_configs(default_config, user_config)
    else:
        if config_path:
            print(f"Config file {config_path} not found, using defaults")
        else:
            print("No config specified, using defaults")
        config = default_config
    
    return config


def _merge_configs(default, user):
    """Recursively merge user configuration with defaults."""
    result = default.copy()
    
    for key, value in user.items():
        if isinstance(value, dict) and key in result and isinstance(result[key], dict):
            result[key] = _merge_configs(result[key], value)
        else:
            result[key] = value
    
    return result


def run_ga_experiment(X, y, dataset_name, config, n_folds=5):
    """Run GA with FAST settings using configuration."""
    print(f"\n{'='*70}")
    print(f"Running FAST GA on {dataset_name}")
    print(f"{'='*70}")
    
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=config['experiment']['random_state'])
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
        
        ga_config = GAConfig(
            population_size=config['ga']['population_size'],
            n_generations=config['ga']['n_generations'],
            crossover_prob=config['ga']['crossover_prob'],
            mutation_prob=config['ga']['mutation_prob'],
            tournament_size=config['ga']['tournament_size'],
            elitism_ratio=config['ga']['elitism_ratio'],
            mutation_types=config['ga']['mutation_types']
        )
        
        initializer = TreeInitializer(
            n_features=n_features,
            n_classes=n_classes,
            max_depth=config['tree']['max_depth'],
            min_samples_split=config['tree']['min_samples_split'],
            min_samples_leaf=config['tree']['min_samples_leaf']
        )
        
        fitness_config = config['fitness']
        fitness_calc = FastFitnessCalculator(
            mode=fitness_config['mode'],
            accuracy_weight=fitness_config['accuracy_weight'],
            interpretability_weight=fitness_config['interpretability_weight'],
            interpretability_weights=fitness_config['interpretability_weights']
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


def run_cart_experiment(X, y, dataset_name, config, n_folds=5):
    """Run CART baseline."""
    print(f"\n{'='*70}")
    print(f"Running CART on {dataset_name}")
    print(f"{'='*70}")
    
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=config['experiment']['random_state'])
    results = {'test_acc': [], 'test_f1': [], 'nodes': [], 'depth': [], 'time': []}
    
    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
        print(f"  Fold {fold}/{n_folds}...", end=' ')
        
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        start = time.time()
        model = DecisionTreeClassifier(
            max_depth=config['tree']['max_depth'], 
            random_state=config['experiment']['random_state']
        )
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


def run_rf_experiment(X, y, dataset_name, config, n_folds=5):
    """Run Random Forest baseline."""
    print(f"\n{'='*70}")
    print(f"Running Random Forest on {dataset_name}")
    print(f"{'='*70}")
    
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=config['experiment']['random_state'])
    results = {'test_acc': [], 'test_f1': [], 'time': []}
    
    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
        print(f"  Fold {fold}/{n_folds}...", end=' ')
        
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        start = time.time()
        model = RandomForestClassifier(
            n_estimators=100, 
            max_depth=config['tree']['max_depth'], 
            random_state=config['experiment']['random_state'], 
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        elapsed = time.time() - start
        
        y_pred = model.predict(X_test)
        
        results['test_acc'].append(accuracy_score(y_test, y_pred))
        results['test_f1'].append(f1_score(y_test, y_pred, average='weighted'))
        results['time'].append(elapsed)
        
        print(f"Acc={results['test_acc'][-1]:.3f}, Time={elapsed:.1f}s")
    
    return results


def print_summary(all_results, config):
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
    
    # Save results
    output_dir = Path('results')
    output_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = output_dir / f'results_FAST_{timestamp}.csv'
    df.to_csv(results_file, index=False)
    
    print(f"\n✓ Results saved to: {results_file}")


def main():
    """Run FAST experiments with configurable parameters."""
    parser = argparse.ArgumentParser(description='Run GA-optimized decision tree experiments')
    parser.add_argument('--config', type=str, help='Path to configuration YAML file')
    parser.add_argument('--datasets', nargs='+', help='List of datasets to use (overrides config)')
    args = parser.parse_args()
    
    config = load_config(args.config)
    
    # Override datasets if specified
    if args.datasets:
        config['experiment']['datasets'] = args.datasets
    
    print(f"\n{'='*70}")
    print("GA-Optimized Decision Trees: Enhanced Dataset Support")
    print(f"{'='*70}")
    
    print("\nConfiguration:")
    print(f"  Datasets: {', '.join(config['experiment']['datasets'])}")
    print(f"  GA: {config['ga']['population_size']} pop, {config['ga']['n_generations']} gen")
    print(f"  Tree: max_depth={config['tree']['max_depth']}")
    
    all_results = {}
    total_start = time.time()
    
    for dataset_name in config['experiment']['datasets']:
        print(f"\n{'='*70}")
        print(f"Loading dataset: {dataset_name}")
        print(f"{'='*70}")
        
        try:
            X, y = load_any_dataset(dataset_name)
            print(f"✓ Loaded: {X.shape[0]} samples, {X.shape[1]} features, {len(np.unique(y))} classes")
            
            dataset_results = {}
            dataset_results['GA-Optimized'] = run_ga_experiment(X, y, dataset_name, config, 
                                                               n_folds=config['experiment']['cv_folds'])
            dataset_results['CART'] = run_cart_experiment(X, y, dataset_name, config, 
                                                         n_folds=config['experiment']['cv_folds'])
            dataset_results['Random Forest'] = run_rf_experiment(X, y, dataset_name, config, 
                                                                n_folds=config['experiment']['cv_folds'])
            
            all_results[dataset_name] = dataset_results
            
        except Exception as e:
            print(f"✗ Failed to process {dataset_name}: {e}")
            import traceback
            traceback.print_exc()
    
    total_time = time.time() - total_start
    
    if all_results:
        print_summary(all_results, config)
    
    print(f"\n{'='*70}")
    print(f"Total Time: {total_time:.1f}s (~{total_time/60:.1f} minutes)")
    print("Experiment Complete! 🎉")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()