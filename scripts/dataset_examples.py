"""
Dataset Loader Examples
Demonstrates various ways to use the enhanced dataset loader

Usage:
    python scripts/dataset_examples.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import numpy as np
import pandas as pd
from ga_trees.data.dataset_loader import DatasetLoader, load_benchmark_dataset


def example_1_basic_loading():
    """Example 1: Basic dataset loading."""
    print("\n" + "="*70)
    print("Example 1: Basic Dataset Loading")
    print("="*70)
    
    # Load iris dataset
    data = load_benchmark_dataset('iris', test_size=0.2)
    
    print(f"\nDataset: Iris")
    print(f"  Training samples: {len(data['X_train'])}")
    print(f"  Test samples: {len(data['X_test'])}")
    print(f"  Features: {data['metadata']['n_features']}")
    print(f"  Classes: {data['metadata']['n_classes']}")
    print(f"  Feature names: {data['feature_names'][:3]}...")
    

def example_2_list_datasets():
    """Example 2: List all available datasets."""
    print("\n" + "="*70)
    print("Example 2: List Available Datasets")
    print("="*70)
    
    available = DatasetLoader.list_available_datasets()
    
    print(f"\nSklearn datasets ({len(available['sklearn'])}):")
    for name in available['sklearn']:
        print(f"  - {name}")
    
    print(f"\nOpenML datasets ({len(available['openml'])}):")
    for name in list(available['openml'])[:10]:  # Show first 10
        print(f"  - {name}")
    print(f"  ... and {len(available['openml']) - 10} more")


def example_3_standardization():
    """Example 3: Load with standardization."""
    print("\n" + "="*70)
    print("Example 3: Feature Standardization")
    print("="*70)
    
    # Load without standardization
    data_raw = load_benchmark_dataset('diabetes_pima', 
                                     test_size=0.2, 
                                     standardize=False)
    
    # Load with standardization
    data_std = load_benchmark_dataset('diabetes_pima', 
                                     test_size=0.2, 
                                     standardize=True)
    
    print(f"\nDiabetes Dataset:")
    print(f"  Original scale - Mean: {data_raw['X_train'][:, 0].mean():.2f}, "
          f"Std: {data_raw['X_train'][:, 0].std():.2f}")
    print(f"  Standardized   - Mean: {data_std['X_train'][:, 0].mean():.2f}, "
          f"Std: {data_std['X_train'][:, 0].std():.2f}")
    
    # Scaler can be used on new data
    scaler = data_std['scaler']
    print(f"\n✓ Scaler saved for transforming new data")


def example_4_imbalanced_data():
    """Example 4: Handle imbalanced datasets."""
    print("\n" + "="*70)
    print("Example 4: Balancing Imbalanced Data")
    print("="*70)
    
    # Load without balancing
    data_orig = load_benchmark_dataset('hepatitis', 
                                      test_size=0.2,
                                      balance=None)
    
    # Count classes in original
    unique_orig, counts_orig = np.unique(data_orig['y_train'], return_counts=True)
    
    print(f"\nHepatitis Dataset (Original):")
    for cls, count in zip(unique_orig, counts_orig):
        print(f"  Class {cls}: {count} samples")
    print(f"  Imbalance ratio: {counts_orig.max() / counts_orig.min():.2f}:1")
    
    # Load with oversampling
    data_over = load_benchmark_dataset('hepatitis',
                                      test_size=0.2,
                                      balance='oversample')
    
    unique_over, counts_over = np.unique(data_over['y_train'], return_counts=True)
    
    print(f"\nHepatitis Dataset (Oversampled):")
    for cls, count in zip(unique_over, counts_over):
        print(f"  Class {cls}: {count} samples")
    print(f"  Imbalance ratio: {counts_over.max() / counts_over.min():.2f}:1")


def example_5_openml_datasets():
    """Example 5: Load various OpenML datasets."""
    print("\n" + "="*70)
    print("Example 5: OpenML Datasets")
    print("="*70)
    
    datasets_to_try = ['titanic', 'heart', 'credit_g', 'sonar']
    
    for name in datasets_to_try:
        try:
            data = load_benchmark_dataset(name, test_size=0.2)
            print(f"\n✓ {name.upper()}")
            print(f"    Samples: {data['metadata']['n_samples']}")
            print(f"    Features: {data['metadata']['n_features']}")
            print(f"    Classes: {data['metadata']['n_classes']}")
        except Exception as e:
            print(f"\n✗ {name}: {e}")


def example_6_csv_loading():
    """Example 6: Load custom CSV file."""
    print("\n" + "="*70)
    print("Example 6: Load Custom CSV File")
    print("="*70)
    
    # Create a sample CSV for demonstration
    sample_csv = Path('data') / 'sample_dataset.csv'
    sample_csv.parent.mkdir(exist_ok=True)
    
    # Generate synthetic data
    np.random.seed(42)
    n_samples = 200
    df = pd.DataFrame({
        'feature1': np.random.randn(n_samples),
        'feature2': np.random.randn(n_samples),
        'feature3': np.random.randn(n_samples),
        'target': np.random.randint(0, 2, n_samples)
    })
    df.to_csv(sample_csv, index=False)
    
    print(f"\nCreated sample CSV: {sample_csv}")
    print(f"  Shape: {df.shape}")
    
    # Load the CSV
    data = load_benchmark_dataset(str(sample_csv), test_size=0.3)
    
    print(f"\nLoaded from CSV:")
    print(f"  Training samples: {len(data['X_train'])}")
    print(f"  Test samples: {len(data['X_test'])}")
    print(f"  Features: {data['feature_names']}")


def example_7_compare_datasets():
    """Example 7: Compare characteristics of multiple datasets."""
    print("\n" + "="*70)
    print("Example 7: Dataset Comparison")
    print("="*70)
    
    datasets = ['iris', 'wine', 'breast_cancer', 'heart']
    
    results = []
    for name in datasets:
        try:
            data = load_benchmark_dataset(name, test_size=0.2)
            meta = data['metadata']
            
            results.append({
                'Dataset': name,
                'Samples': meta['n_samples'],
                'Features': meta['n_features'],
                'Classes': meta['n_classes']
            })
        except:
            pass
    
    df = pd.DataFrame(results)
    print("\n" + df.to_string(index=False))


def example_8_validation():
    """Example 8: Data validation in action."""
    print("\n" + "="*70)
    print("Example 8: Automatic Data Validation")
    print("="*70)
    
    # Create problematic data
    X = np.array([
        [1.0, 2.0, 3.0],
        [4.0, np.nan, 6.0],  # NaN value
        [7.0, 8.0, 9.0],
        [10.0, 11.0, 12.0]
    ])
    y = np.array([0, 0, 0, 1])  # Imbalanced
    
    from ga_trees.data.dataset_loader import DataValidator
    
    validator = DataValidator()
    
    print("\nValidating problematic dataset...")
    valid, warnings = validator.validate_dataset(X, y)
    
    print(f"  Valid: {valid}")
    if warnings:
        print("  Warnings:")
        for w in warnings:
            print(f"    - {w}")
    
    # Clean the data
    X_clean, y_clean = validator.clean_dataset(X, y, strategy='mean')
    print(f"\n✓ Data cleaned (NaN replaced with column mean)")
    print(f"  Original shape: {X.shape}")
    print(f"  Cleaned shape: {X_clean.shape}")


def example_9_integration_with_ga():
    """Example 9: Full integration with GA training."""
    print("\n" + "="*70)
    print("Example 9: Integration with GA Training")
    print("="*70)
    
    # Load dataset
    data = load_benchmark_dataset('heart', 
                                 test_size=0.2, 
                                 standardize=True)
    
    X_train = data['X_train']
    y_train = data['y_train']
    X_test = data['X_test']
    y_test = data['y_test']
    
    print(f"\nLoaded Heart Disease dataset")
    print(f"  Training: {len(X_train)} samples")
    print(f"  Test: {len(X_test)} samples")
    
    # Setup GA (simplified for demo)
    from ga_trees.ga.engine import GAEngine, GAConfig, TreeInitializer, Mutation
    from ga_trees.fitness.calculator import FitnessCalculator, TreePredictor
    
    n_features = data['metadata']['n_features']
    n_classes = data['metadata']['n_classes']
    feature_ranges = {i: (X_train[:, i].min(), X_train[:, i].max()) 
                     for i in range(n_features)}
    
    ga_config = GAConfig(population_size=30, n_generations=10)
    initializer = TreeInitializer(n_features=n_features, n_classes=n_classes,
                                 max_depth=5, min_samples_split=5, min_samples_leaf=2)
    fitness_calc = FitnessCalculator()
    mutation = Mutation(n_features=n_features, feature_ranges=feature_ranges)
    
    print(f"\nTraining GA (30 pop, 10 gen)...")
    ga_engine = GAEngine(ga_config, initializer, 
                        fitness_calc.calculate_fitness, mutation)
    best_tree = ga_engine.evolve(X_train, y_train, verbose=False)
    
    # Evaluate
    predictor = TreePredictor()
    y_pred = predictor.predict(best_tree, X_test)
    
    from sklearn.metrics import accuracy_score
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\n✓ Training complete!")
    print(f"  Test Accuracy: {accuracy:.4f}")
    print(f"  Tree Size: {best_tree.get_num_nodes()} nodes")
    print(f"  Tree Depth: {best_tree.get_depth()}")


def main():
    """Run all examples."""
    print("\n" + "="*70)
    print("GA-Trees Dataset Loader Examples")
    print("="*70)
    
    examples = [
        ("Basic Loading", example_1_basic_loading),
        ("List Datasets", example_2_list_datasets),
        ("Standardization", example_3_standardization),
        ("Imbalanced Data", example_4_imbalanced_data),
        ("OpenML Datasets", example_5_openml_datasets),
        ("CSV Loading", example_6_csv_loading),
        ("Dataset Comparison", example_7_compare_datasets),
        ("Data Validation", example_8_validation),
        ("GA Integration", example_9_integration_with_ga)
    ]
    
    print("\nAvailable examples:")
    for i, (name, _) in enumerate(examples, 1):
        print(f"  {i}. {name}")
    
    print("\nRunning all examples...\n")
    
    for name, example_func in examples:
        try:
            example_func()
        except Exception as e:
            print(f"\n✗ {name} failed: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*70)
    print("Examples Complete!")
    print("="*70)


if __name__ == '__main__':
    main()