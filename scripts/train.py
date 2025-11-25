"""
Training script for GA-optimized decision trees.

NOW SUPPORTS CONFIG FILES!

Usage:
    # With config file (recommended)
    python scripts/train.py --config configs/custom.yaml --dataset breast_cancer
    
    # With command-line args (old way still works)
    python scripts/train.py --dataset iris --generations 50 --population 100
    
    # Mix both (CLI args override config)
    python scripts/train.py --config configs/custom.yaml --generations 60
"""

import argparse
import numpy as np
import pickle
import yaml
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from sklearn.datasets import load_iris, load_wine, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Import our modules
from ga_trees.genotype.tree_genotype import TreeGenotype
from ga_trees.ga.engine import GAEngine, GAConfig, TreeInitializer, Mutation
from ga_trees.fitness.calculator import FitnessCalculator


def load_dataset(name: str):
    """Load dataset by name."""
    if name == 'iris':
        return load_iris(return_X_y=True)
    elif name == 'wine':
        return load_wine(return_X_y=True)
    elif name == 'breast_cancer':
        return load_breast_cancer(return_X_y=True)
    else:
        raise ValueError(f"Unknown dataset: {name}")


def get_feature_ranges(X: np.ndarray) -> dict:
    """Get min/max range for each feature."""
    ranges = {}
    for i in range(X.shape[1]):
        ranges[i] = (float(X[:, i].min()), float(X[:, i].max()))
    return ranges


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def merge_config_with_args(config, args):
    """
    Merge config file with command-line arguments.
    CLI arguments take precedence (if explicitly set by user).
    """
    # Create a dict of "was this arg explicitly set?"
    # If user didn't specify, use config value
    
    if config is None:
        return args  # No config, use all CLI args
    
    # GA parameters
    if args.population == 100:  # Default value = not set by user
        args.population = config['ga']['population_size']
    if args.generations == 50:
        args.generations = config['ga']['n_generations']
    if args.crossover_prob == 0.7:
        args.crossover_prob = config['ga']['crossover_prob']
    if args.mutation_prob == 0.2:
        args.mutation_prob = config['ga']['mutation_prob']
    if args.tournament_size == 3:
        args.tournament_size = config['ga']['tournament_size']
    if args.elitism_ratio == 0.1:
        args.elitism_ratio = config['ga']['elitism_ratio']
    
    # Tree parameters
    if args.max_depth == 5:
        args.max_depth = config['tree']['max_depth']
    if args.min_samples_split == 10:
        args.min_samples_split = config['tree']['min_samples_split']
    if args.min_samples_leaf == 5:
        args.min_samples_leaf = config['tree']['min_samples_leaf']
    
    # Fitness parameters
    if args.accuracy_weight == 0.7:
        args.accuracy_weight = config['fitness']['weights']['accuracy']
    if args.interpretability_weight == 0.3:
        args.interpretability_weight = config['fitness']['weights']['interpretability']
    
    return args


def main():
    parser = argparse.ArgumentParser(description='Train GA-optimized decision tree')
    
    # NEW: Config file support
    parser.add_argument('--config', type=str, default=None,
                       help='Path to YAML config file (recommended)')
    
    # Dataset args
    parser.add_argument('--dataset', type=str, default='iris',
                       choices=['iris', 'wine', 'breast_cancer'],
                       help='Dataset to use')
    parser.add_argument('--test-size', type=float, default=0.2,
                       help='Test set size (0-1)')
    parser.add_argument('--standardize', action='store_true',
                       help='Standardize features')
    
    # GA args
    parser.add_argument('--population', type=int, default=100,
                       help='Population size')
    parser.add_argument('--generations', type=int, default=50,
                       help='Number of generations')
    parser.add_argument('--crossover-prob', type=float, default=0.7,
                       help='Crossover probability')
    parser.add_argument('--mutation-prob', type=float, default=0.2,
                       help='Mutation probability')
    parser.add_argument('--tournament-size', type=int, default=3,
                       help='Tournament size')
    parser.add_argument('--elitism-ratio', type=float, default=0.1,
                       help='Elitism ratio')
    
    # Tree args
    parser.add_argument('--max-depth', type=int, default=5,
                       help='Maximum tree depth')
    parser.add_argument('--min-samples-split', type=int, default=10,
                       help='Minimum samples to split')
    parser.add_argument('--min-samples-leaf', type=int, default=5,
                       help='Minimum samples in leaf')
    
    # Fitness args
    parser.add_argument('--accuracy-weight', type=float, default=0.7,
                       help='Weight for accuracy in fitness')
    parser.add_argument('--interpretability-weight', type=float, default=0.3,
                       help='Weight for interpretability in fitness')
    
    # Output args
    parser.add_argument('--output', type=str, default='models/best_tree.pkl',
                       help='Output file for best model')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--verbose', action='store_true',
                       help='Print progress')
    
    args = parser.parse_args()
    
    # Load and merge config if provided
    config = None
    if args.config:
        print(f"Loading configuration from: {args.config}")
        config = load_config(args.config)
        args = merge_config_with_args(config, args)
        print("✓ Config loaded and merged with CLI arguments")
    
    # Set random seeds
    np.random.seed(args.seed)
    import random
    random.seed(args.seed)
    
    print(f"\nTraining GA-optimized decision tree on {args.dataset}")
    print(f"Configuration: pop={args.population}, gen={args.generations}, "
          f"depth={args.max_depth}")
    if args.config:
        print(f"Using config file: {args.config}")
    
    # Load data
    X, y = load_dataset(args.dataset)
    n_features = X.shape[1]
    n_classes = len(np.unique(y))
    
    print(f"Dataset: {X.shape[0]} samples, {n_features} features, {n_classes} classes")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.seed, stratify=y
    )
    
    # Standardize if requested
    scaler = None
    if args.standardize:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
    
    print(f"Train: {len(X_train)} samples, Test: {len(X_test)} samples")
    
    # Get feature ranges
    feature_ranges = get_feature_ranges(X_train)
    
    # Create components
    mutation_types = config['ga']['mutation_types'] if config else {
        'threshold_perturbation': 0.5,
        'feature_replacement': 0.3,
        'prune_subtree': 0.15,
        'expand_leaf': 0.05
    }
    
    ga_config = GAConfig(
        population_size=args.population,
        n_generations=args.generations,
        crossover_prob=args.crossover_prob,
        mutation_prob=args.mutation_prob,
        tournament_size=args.tournament_size,
        elitism_ratio=args.elitism_ratio,
        mutation_types=mutation_types
    )
    
    initializer = TreeInitializer(
        n_features=n_features,
        n_classes=n_classes,
        max_depth=args.max_depth,
        min_samples_split=args.min_samples_split,
        min_samples_leaf=args.min_samples_leaf,
        task_type='classification'
    )
    
    # Get interpretability weights if available
    interp_weights = None
    if config and 'interpretability_weights' in config['fitness']:
        interp_weights = config['fitness']['interpretability_weights']
    
    fitness_calc = FitnessCalculator(
        mode='weighted_sum',
        accuracy_weight=args.accuracy_weight,
        interpretability_weight=args.interpretability_weight,
        interpretability_weights=interp_weights
    )
    
    mutation = Mutation(
        n_features=n_features,
        feature_ranges=feature_ranges
    )
    
    # Create GA engine
    ga_engine = GAEngine(
        config=ga_config,
        initializer=initializer,
        fitness_function=fitness_calc.calculate_fitness,
        mutation=mutation
    )
    
    # Run evolution
    print("\nStarting evolution...")
    best_tree = ga_engine.evolve(X_train, y_train, verbose=args.verbose)
    
    print("\n" + "="*60)
    print("Evolution complete!")
    print("="*60)
    
    # Evaluate on test set
    from ga_trees.fitness.calculator import TreePredictor
    predictor = TreePredictor()
    
    y_train_pred = predictor.predict(best_tree, X_train)
    y_test_pred = predictor.predict(best_tree, X_test)
    
    from sklearn.metrics import accuracy_score, f1_score
    
    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)
    test_f1 = f1_score(y_test, y_test_pred, average='weighted')
    
    print(f"\nBest Tree Statistics:")
    print(f"  Depth: {best_tree.get_depth()}")
    print(f"  Nodes: {best_tree.get_num_nodes()}")
    print(f"  Leaves: {best_tree.get_num_leaves()}")
    print(f"  Features Used: {best_tree.get_num_features_used()}/{n_features}")
    print(f"  Tree Balance: {best_tree.get_tree_balance():.4f}")
    
    print(f"\nPerformance:")
    print(f"  Train Accuracy: {train_acc:.4f}")
    print(f"  Test Accuracy:  {test_acc:.4f}")
    print(f"  Test F1 Score:  {test_f1:.4f}")
    print(f"  Fitness Score:  {best_tree.fitness_:.4f}")
    print(f"  Interpretability: {best_tree.interpretability_:.4f}")
    
    # Print rules
    print(f"\nDecision Rules:")
    rules = best_tree.to_rules()
    for i, rule in enumerate(rules[:10], 1):  # Show first 10 rules
        print(f"  {i}. {rule}")
    if len(rules) > 10:
        print(f"  ... ({len(rules)-10} more rules)")
    
    # Save model
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    model_data = {
        'tree': best_tree,
        'config': vars(args),
        'config_file': args.config,
        'feature_ranges': feature_ranges,
        'n_features': n_features,
        'n_classes': n_classes,
        'scaler': scaler if args.standardize else None,
        'metrics': {
            'train_accuracy': train_acc,
            'test_accuracy': test_acc,
            'test_f1': test_f1,
        }
    }
    
    with open(output_path, 'wb') as f:
        pickle.dump(model_data, f)
    
    print(f"\n✓ Model saved to: {output_path}")
    
    # Plot evolution history
    try:
        import matplotlib.pyplot as plt
        
        history = ga_engine.get_history()
        
        plt.figure(figsize=(10, 6))
        plt.plot(history['best_fitness'], label='Best Fitness', linewidth=2)
        plt.plot(history['avg_fitness'], label='Average Fitness', linewidth=2, alpha=0.7)
        plt.xlabel('Generation')
        plt.ylabel('Fitness')
        plt.title(f'GA Evolution - {args.dataset}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plot_path = output_path.parent / f"{output_path.stem}_evolution.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"✓ Evolution plot saved to: {plot_path}")
        
    except ImportError:
        print("Matplotlib not available, skipping evolution plot")
    
    print("\n" + "="*60)
    print("Training Complete! 🎉")
    print("="*60)


if __name__ == '__main__':
    main()