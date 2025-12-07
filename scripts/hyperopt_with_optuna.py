"""
Complete Hyperparameter Optimization with Optuna
Implements all requirements: Bayesian optimization, early stopping, presets, tracking

Usage:
    # Basic optimization
    python scripts/hyperopt_with_optuna.py --dataset breast_cancer --n-trials 50
    
    # With preset
    python scripts/hyperopt_with_optuna.py --preset fast --dataset iris
    
    # With early stopping
    python scripts/hyperopt_with_optuna.py --n-trials 100 --early-stopping 10
    
    # Resume previous study
    python scripts/hyperopt_with_optuna.py --study-name my_study --resume
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import argparse
import optuna
import numpy as np
import yaml
import json
import logging
from datetime import datetime
from typing import Dict, Any, Optional
from sklearn.datasets import load_breast_cancer, load_iris, load_wine
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score

from ga_trees.ga.engine import GAEngine, GAConfig, TreeInitializer, Mutation
from ga_trees.fitness.calculator import FitnessCalculator, TreePredictor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# OPTIMIZATION PRESETS - Task: Create optimization presets for common use cases
# ============================================================================

OPTIMIZATION_PRESETS = {
    'fast': {
        'description': 'Quick optimization (10 trials, small search space)',
        'n_trials': 10,
        'timeout': 600,  # 10 minutes
        'cv_folds': 3,
        'search_space': {
            'population_size': {'type': 'categorical', 'choices': [50, 80, 100]},
            'n_generations': {'type': 'categorical', 'choices': [20, 30, 40]},
            'crossover_prob': {'type': 'float', 'low': 0.6, 'high': 0.8},
            'mutation_prob': {'type': 'float', 'low': 0.1, 'high': 0.25},
            'accuracy_weight': {'type': 'float', 'low': 0.6, 'high': 0.75}
        }
    },
    
    'balanced': {
        'description': 'Balanced optimization (30 trials, medium search space)',
        'n_trials': 30,
        'timeout': 1800,  # 30 minutes
        'cv_folds': 5,
        'search_space': {
            'population_size': {'type': 'int', 'low': 50, 'high': 150, 'step': 10},
            'n_generations': {'type': 'int', 'low': 20, 'high': 60, 'step': 10},
            'crossover_prob': {'type': 'float', 'low': 0.5, 'high': 0.9},
            'mutation_prob': {'type': 'float', 'low': 0.1, 'high': 0.3},
            'tournament_size': {'type': 'int', 'low': 2, 'high': 5},
            'max_depth': {'type': 'int', 'low': 3, 'high': 7},
            'accuracy_weight': {'type': 'float', 'low': 0.6, 'high': 0.9},
            'node_complexity_weight': {'type': 'float', 'low': 0.4, 'high': 0.7}
        }
    },
    
    'thorough': {
        'description': 'Thorough optimization (100 trials, full search space)',
        'n_trials': 100,
        'timeout': 7200,  # 2 hours
        'cv_folds': 5,
        'search_space': {
            'population_size': {'type': 'int', 'low': 50, 'high': 200, 'step': 10},
            'n_generations': {'type': 'int', 'low': 30, 'high': 60, 'step': 10},
            'crossover_prob': {'type': 'float', 'low': 0.5, 'high': 0.95},
            'mutation_prob': {'type': 'float', 'low': 0.05, 'high': 0.35},
            'tournament_size': {'type': 'int', 'low': 2, 'high': 6},
            'elitism_ratio': {'type': 'float', 'low': 0.05, 'high': 0.2},
            'max_depth': {'type': 'int', 'low': 5, 'high': 7},
            'min_samples_split': {'type': 'int', 'low': 5, 'high': 20},
            'min_samples_leaf': {'type': 'int', 'low': 2, 'high': 10},
            'accuracy_weight': {'type': 'float', 'low': 0.60, 'high': 0.75},
            'node_complexity_weight': {'type': 'float', 'low': 0.3, 'high': 0.8},
            'feature_coherence_weight': {'type': 'float', 'low': 0.05, 'high': 0.35}
        }
    },
    
    'interpretability_focused': {
        'description': 'Focus on small trees with good accuracy',
        'n_trials': 50,
        'timeout': 3600,  # 1 hour
        'cv_folds': 5,
        'search_space': {
            'population_size': {'type': 'int', 'low': 50, 'high': 120, 'step': 10},
            'n_generations': {'type': 'int', 'low': 30, 'high': 60, 'step': 10},
            'accuracy_weight': {'type': 'float', 'low': 0.5, 'high': 0.75},
            'node_complexity_weight': {'type': 'float', 'low': 0.5, 'high': 0.8},
            'feature_coherence_weight': {'type': 'float', 'low': 0.15, 'high': 0.35},
            'max_depth': {'type': 'int', 'low': 3, 'high': 6}
        }
    },
    
    'accuracy_focused': {
        'description': 'Maximize accuracy with less interpretability constraint',
        'n_trials': 50,
        'timeout': 3600,
        'cv_folds': 5,
        'search_space': {
            'population_size': {'type': 'int', 'low': 80, 'high': 180, 'step': 20},
            'n_generations': {'type': 'int', 'low': 40, 'high': 80, 'step': 10},
            'accuracy_weight': {'type': 'float', 'low': 0.75, 'high': 0.95},
            'node_complexity_weight': {'type': 'float', 'low': 0.2, 'high': 0.5},
            'max_depth': {'type': 'int', 'low': 5, 'high': 8}
        }
    }
}


# ============================================================================
# BAYESIAN OPTIMIZATION WITH PRUNING - Tasks: Bayesian strategies, early stopping
# ============================================================================

class OptimizationTracker:
    """Track and log optimization progress."""
    
    def __init__(self, study_name: str, output_dir: str = 'results/optimization'):
        self.study_name = study_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.history = {
            'trials': [],
            'best_values': [],
            'best_params': [],
            'timestamps': []
        }
        
        self.start_time = datetime.now()
    
    def log_trial(self, trial: optuna.Trial, value: float, params: Dict):
        """Log trial results."""
        self.history['trials'].append(trial.number)
        self.history['best_values'].append(value)
        self.history['best_params'].append(params)
        self.history['timestamps'].append(datetime.now().isoformat())
        
        # Save progress periodically
        if trial.number % 10 == 0:
            self.save_progress()
    
    def save_progress(self):
        """Save optimization progress to JSON."""
        progress_file = self.output_dir / f'{self.study_name}_progress.json'
        
        with open(progress_file, 'w') as f:
            json.dump({
                'study_name': self.study_name,
                'start_time': self.start_time.isoformat(),
                'current_time': datetime.now().isoformat(),
                'n_trials': len(self.history['trials']),
                'history': self.history
            }, f, indent=2)
        
        logger.info(f"Progress saved to {progress_file}")
    
    def save_final_report(self, study: optuna.Study):
        """Save final optimization report."""
        report_file = self.output_dir / f'{self.study_name}_report.json'
        
        report = {
            'study_name': self.study_name,
            'start_time': self.start_time.isoformat(),
            'end_time': datetime.now().isoformat(),
            'duration_seconds': (datetime.now() - self.start_time).total_seconds(),
            'n_trials': len(study.trials),
            'best_value': study.best_value,
            'best_params': study.best_params,
            'best_trial_number': study.best_trial.number,
            'all_trials': [
                {
                    'number': t.number,
                    'value': t.value,
                    'params': t.params,
                    'state': str(t.state),
                    'duration': t.duration.total_seconds() if t.duration else None
                }
                for t in study.trials
            ]
        }
        
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Final report saved to {report_file}")


def load_dataset(name: str):
    """Load dataset by name."""
    datasets = {
        'iris': load_iris,
        'wine': load_wine,
        'breast_cancer': load_breast_cancer
    }
    return datasets[name](return_X_y=True)


def suggest_parameter(trial: optuna.Trial, name: str, spec: Dict) -> Any:
    """Suggest parameter based on specification."""
    param_type = spec['type']
    
    if param_type == 'int':
        return trial.suggest_int(name, spec['low'], spec['high'], 
                                step=spec.get('step', 1))
    elif param_type == 'float':
        return trial.suggest_float(name, spec['low'], spec['high'])
    elif param_type == 'categorical':
        return trial.suggest_categorical(name, spec['choices'])
    else:
        raise ValueError(f"Unknown parameter type: {param_type}")


def objective(trial: optuna.Trial, 
              dataset_name: str,
              search_space: Dict,
              cv_folds: int,
              tracker: OptimizationTracker) -> float:
    """
    Optuna objective function with Bayesian optimization.
    
    Task: Implement Bayesian optimization strategies
    """
    # Load data
    X, y = load_dataset(dataset_name)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # Suggest hyperparameters based on search space
    params = {}
    for param_name, spec in search_space.items():
        params[param_name] = suggest_parameter(trial, param_name, spec)
    
    # Default values for parameters not in search space
    population_size = params.get('population_size', 80)
    n_generations = params.get('n_generations', 40)
    crossover_prob = params.get('crossover_prob', 0.72)
    mutation_prob = params.get('mutation_prob', 0.18)
    tournament_size = params.get('tournament_size', 4)
    elitism_ratio = params.get('elitism_ratio', 0.12)
    max_depth = params.get('max_depth', 6)
    min_samples_split = params.get('min_samples_split', 8)
    min_samples_leaf = params.get('min_samples_leaf', 3)
    accuracy_weight = params.get('accuracy_weight', 0.68)
    
    # Interpretability sub-weights
    node_complexity_weight = params.get('node_complexity_weight', 0.50)
    feature_coherence_weight = params.get('feature_coherence_weight', 0.10)
    
    # Normalize interpretability weights
    remaining_weight = 1.0 - node_complexity_weight - feature_coherence_weight
    tree_balance_weight = remaining_weight * 0.33
    semantic_coherence_weight = remaining_weight * 0.67
    
    # Setup
    n_features = X.shape[1]
    n_classes = len(np.unique(y))
    
    ga_config = GAConfig(
        population_size=population_size,
        n_generations=n_generations,
        crossover_prob=crossover_prob,
        mutation_prob=mutation_prob,
        tournament_size=tournament_size,
        elitism_ratio=elitism_ratio
    )
    
    # Train and evaluate with CV
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    scores = []
    tree_sizes = []
    
    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        feature_ranges = {j: (X_train[:, j].min(), X_train[:, j].max()) 
                         for j in range(n_features)}
        
        initializer = TreeInitializer(
            n_features=n_features,
            n_classes=n_classes,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf
        )
        
        fitness_calc = FitnessCalculator(
            mode='weighted_sum',
            accuracy_weight=accuracy_weight,
            interpretability_weight=1.0 - accuracy_weight,
            interpretability_weights={
                'node_complexity': node_complexity_weight,
                'feature_coherence': feature_coherence_weight,
                'tree_balance': tree_balance_weight,
                'semantic_coherence': semantic_coherence_weight
            }
        )
        
        mutation = Mutation(n_features=n_features, feature_ranges=feature_ranges)
        
        try:
            ga_engine = GAEngine(ga_config, initializer, 
                               fitness_calc.calculate_fitness, mutation)
            best_tree = ga_engine.evolve(X_train, y_train, verbose=False)
            
            predictor = TreePredictor()
            y_pred = predictor.predict(best_tree, X_test)
            
            acc = accuracy_score(y_test, y_pred)
            scores.append(acc)
            tree_sizes.append(best_tree.get_num_nodes())
            
        except Exception as e:
            logger.warning(f"Trial {trial.number} fold {fold} failed: {e}")
            return 0.0
        
        # Task: Implement early stopping and pruning
        # Report intermediate value for pruning
        if fold < cv_folds - 1:
            intermediate_score = np.mean(scores)
            trial.report(intermediate_score, fold)
            
            # Handle pruning
            if trial.should_prune():
                logger.info(f"Trial {trial.number} pruned at fold {fold}")
                raise optuna.TrialPruned()
    
    # Calculate final score (combine accuracy and tree size)
    mean_accuracy = np.mean(scores)
    mean_tree_size = np.mean(tree_sizes)
    
    # Composite score: 80% accuracy + 20% size penalty
    max_tree_size = 50  # Normalize by maximum expected tree size
    size_penalty = 1.0 - min(mean_tree_size / max_tree_size, 1.0)
    final_score = mean_accuracy
    if mean_tree_size > 50:
       final_score -= min((mean_tree_size - 50) / 100, 0.05)
    
    # Log trial
    tracker.log_trial(trial, final_score, params)
    
    # Log to Optuna's trial user attributes
    trial.set_user_attr('mean_accuracy', mean_accuracy)
    trial.set_user_attr('mean_tree_size', mean_tree_size)
    trial.set_user_attr('std_accuracy', np.std(scores))
    
    logger.info(f"Trial {trial.number}: Score={final_score:.4f}, "
                f"Accuracy={mean_accuracy:.4f}, TreeSize={mean_tree_size:.1f}")
    
    return final_score


def create_study(study_name: str, 
                 storage: Optional[str] = None,
                 resume: bool = False) -> optuna.Study:
    """
    Create or load Optuna study with Bayesian optimization.
    
    Task: Add Bayesian optimization strategies
    """
    # Use TPE (Tree-structured Parzen Estimator) for Bayesian optimization
    sampler = optuna.samplers.TPESampler(
        seed=42,
        n_startup_trials=10,  # Random search for first 10 trials
        multivariate=True,     # Consider parameter interactions
        constant_liar=True     # Support for parallel optimization
    )
    
    # Task: Implement early stopping and pruning
    # Use MedianPruner to stop unpromising trials early
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=5,    # Don't prune first 5 trials
        n_warmup_steps=2,      # Wait for 2 folds before pruning
        interval_steps=1       # Check after each fold
    )
    
    # Create or load study
    if storage:
        study = optuna.create_study(
            study_name=study_name,
            storage=storage,
            direction='maximize',
            sampler=sampler,
            pruner=pruner,
            load_if_exists=resume
        )
    else:
        study = optuna.create_study(
            study_name=study_name,
            direction='maximize',
            sampler=sampler,
            pruner=pruner
        )
    
    return study


def save_best_config(study: optuna.Study, 
                    output_path: str = 'configs/optimized.yaml'):
    """Save best parameters to YAML config."""
    best_params = study.best_params
    
    # Convert to config format
    config = {
        'ga': {
            'population_size': best_params.get('population_size', 80),
            'n_generations': best_params.get('n_generations', 40),
            'crossover_prob': best_params.get('crossover_prob', 0.72),
            'mutation_prob': best_params.get('mutation_prob', 0.18),
            'tournament_size': best_params.get('tournament_size', 4),
            'elitism_ratio': best_params.get('elitism_ratio', 0.12)
        },
        'tree': {
            'max_depth': best_params.get('max_depth', 6),
            'min_samples_split': best_params.get('min_samples_split', 8),
            'min_samples_leaf': best_params.get('min_samples_leaf', 3)
        },
        'fitness': {
            'mode': 'weighted_sum',
            'weights': {
                'accuracy': best_params.get('accuracy_weight', 0.68),
                'interpretability': 1.0 - best_params.get('accuracy_weight', 0.68)
            },
            'interpretability_weights': {
                'node_complexity': best_params.get('node_complexity_weight', 0.50),
                'feature_coherence': best_params.get('feature_coherence_weight', 0.10),
                'tree_balance': 0.10,
                'semantic_coherence': 0.30
            }
        },
        'optimization': {
            'study_name': study.study_name,
            'n_trials': len(study.trials),
            'best_value': float(study.best_value),
            'best_trial_number': study.best_trial.number,
            'optimization_date': datetime.now().isoformat()
        }
    }
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    logger.info(f"✓ Best config saved to: {output_path}")


def visualize_optimization(study: optuna.Study, output_dir: str = 'results/figures'):
    """Create optimization visualizations."""
    try:
        import matplotlib.pyplot as plt
        from optuna.visualization import (
            plot_optimization_history,
            plot_param_importances,
            plot_parallel_coordinate
        )
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Optimization history
        fig = plot_optimization_history(study)
        fig.write_html(output_dir / 'optimization_history.html')
        logger.info("✓ Saved: optimization_history.html")
        
        # Parameter importance
        fig = plot_param_importances(study)
        fig.write_html(output_dir / 'param_importance.html')
        logger.info("✓ Saved: param_importance.html")
        
        # Parallel coordinate plot
        try:
            fig = plot_parallel_coordinate(study)
            fig.write_html(output_dir / 'parallel_coordinate.html')
            logger.info("✓ Saved: parallel_coordinate.html")
        except:
            logger.warning("Parallel coordinate plot failed (needs 2+ parameters)")
            
    except ImportError:
        logger.warning("Plotly not installed, skipping visualizations")


def main():
    parser = argparse.ArgumentParser(
        description='Hyperparameter optimization with Optuna',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Fast optimization on Iris
  python scripts/hyperopt_with_optuna.py --preset fast --dataset iris
  
  # Balanced optimization on Breast Cancer
  python scripts/hyperopt_with_optuna.py --preset balanced --dataset breast_cancer
  
  # Custom optimization
  python scripts/hyperopt_with_optuna.py --n-trials 50 --dataset wine --cv-folds 5
  
  # Resume previous study
  python scripts/hyperopt_with_optuna.py --study-name my_study --resume
        """
    )
    
    # Preset selection
    parser.add_argument('--preset', type=str, choices=list(OPTIMIZATION_PRESETS.keys()),
                       help='Use optimization preset')
    
    # Dataset
    parser.add_argument('--dataset', type=str, default='breast_cancer',
                       choices=['iris', 'wine', 'breast_cancer'],
                       help='Dataset to optimize on')
    
    # Optimization parameters
    parser.add_argument('--n-trials', type=int, help='Number of trials (overrides preset)')
    parser.add_argument('--timeout', type=int, help='Timeout in seconds (overrides preset)')
    parser.add_argument('--cv-folds', type=int, help='CV folds (overrides preset)')
    
    # Study management
    parser.add_argument('--study-name', type=str, help='Study name for tracking')
    parser.add_argument('--storage', type=str, help='Optuna storage URL (e.g., sqlite:///optuna.db)')
    parser.add_argument('--resume', action='store_true', help='Resume previous study')
    
    # Output
    parser.add_argument('--output', type=str, default='configs/optimized.yaml',
                       help='Output config file path')
    
    args = parser.parse_args()
    
    # Load preset or use custom settings
    if args.preset:
        preset = OPTIMIZATION_PRESETS[args.preset]
        n_trials = args.n_trials or preset['n_trials']
        timeout = args.timeout or preset['timeout']
        cv_folds = args.cv_folds or preset['cv_folds']
        search_space = preset['search_space']
        
        logger.info(f"Using preset: {args.preset}")
        logger.info(f"Description: {preset['description']}")
    else:
        n_trials = args.n_trials or 30
        timeout = args.timeout or 1800
        cv_folds = args.cv_folds or 5
        search_space = OPTIMIZATION_PRESETS['balanced']['search_space']
    
    # Generate study name
    if not args.study_name:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        study_name = f"ga_opt_{args.dataset}_{timestamp}"
    else:
        study_name = args.study_name
    
    logger.info("="*70)
    logger.info("HYPERPARAMETER OPTIMIZATION WITH OPTUNA")
    logger.info("="*70)
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Study name: {study_name}")
    logger.info(f"Trials: {n_trials}")
    logger.info(f"Timeout: {timeout}s ({timeout/60:.1f} min)")
    logger.info(f"CV folds: {cv_folds}")
    logger.info("="*70)
    
    # Create tracker
    tracker = OptimizationTracker(study_name)
    
    # Create study
    study = create_study(study_name, storage=args.storage, resume=args.resume)
    
    # Optimize
    logger.info("\nStarting optimization...")
    try:
        study.optimize(
            lambda trial: objective(trial, args.dataset, search_space, cv_folds, tracker),
            n_trials=n_trials,
            timeout=timeout,
            show_progress_bar=True,
            catch=(Exception,)  # Catch exceptions and continue
        )
    except KeyboardInterrupt:
        logger.info("\n\nOptimization interrupted by user.")
    
    # Save results
    tracker.save_final_report(study)
    save_best_config(study, args.output)
    visualize_optimization(study)
    
    # Print summary
    logger.info("\n" + "="*70)
    logger.info("OPTIMIZATION COMPLETE")
    logger.info("="*70)
    logger.info(f"\nBest Trial: #{study.best_trial.number}")
    logger.info(f"Best Value: {study.best_value:.4f}")
    logger.info(f"\nBest Parameters:")
    for key, value in study.best_params.items():
        logger.info(f"  {key:25s}: {value}")
    
    # Get user attributes
    best_trial = study.best_trial
    if 'mean_accuracy' in best_trial.user_attrs:
        logger.info(f"\nPerformance Metrics:")
        logger.info(f"  Mean Accuracy: {best_trial.user_attrs['mean_accuracy']:.4f}")
        logger.info(f"  Std Accuracy:  {best_trial.user_attrs['std_accuracy']:.4f}")
        logger.info(f"  Mean Tree Size: {best_trial.user_attrs['mean_tree_size']:.1f}")
    
    logger.info(f"\n✓ Optimized config saved to: {args.output}")
    logger.info(f"✓ Optimization report saved to: results/optimization/{study_name}_report.json")
    logger.info("✓ Visualizations saved to: results/figures/")
    
    logger.info("\n" + "="*70)
    logger.info("Next steps:")
    logger.info(f"1. Review: results/optimization/{study_name}_report.json")
    logger.info(f"2. Test config: python scripts/experiment.py --config {args.output}")
    logger.info(f"3. Compare: python scripts/test_optimized_config.py")
    logger.info("="*70)


if __name__ == '__main__':
    main()