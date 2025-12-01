"""
Quick test to verify the dataset loader fix works.

Run this to confirm all datasets load correctly:
    python scripts/test_dataset_fix.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import numpy as np
from ga_trees.data.dataset_loader import DatasetLoader, load_benchmark_dataset


def test_sklearn_datasets():
    """Test built-in sklearn datasets."""
    print("\n" + "="*70)
    print("Testing Sklearn Datasets")
    print("="*70)
    
    datasets = ['iris', 'wine', 'breast_cancer']
    
    for name in datasets:
        try:
            X, y = load_benchmark_dataset(name, test_size=0.2)['X_train'], \
                   load_benchmark_dataset(name, test_size=0.2)['y_train']
            print(f"✓ {name:20s} - {X.shape[0]} samples, {X.shape[1]} features")
        except Exception as e:
            print(f"✗ {name:20s} - FAILED: {e}")


def test_openml_datasets():
    """Test OpenML datasets (the ones that were broken)."""
    print("\n" + "="*70)
    print("Testing OpenML Datasets (Previously Broken)")
    print("="*70)
    
    datasets = ['heart', 'titanic', 'credit_g', 'sonar', 'hepatitis']
    
    for name in datasets:
        try:
            data = load_benchmark_dataset(name, test_size=0.2)
            X = data['X_train']
            y = data['y_train']
            
            # Check data types
            assert X.dtype in [np.float64, np.float32], f"X is not numeric: {X.dtype}"
            assert y.dtype in [np.int64, np.int32, np.int8], f"y is not integer: {y.dtype}"
            
            # Check for NaN
            assert not np.any(np.isnan(X)), "X contains NaN"
            
            print(f"✓ {name:20s} - {len(X)} samples, {X.shape[1]} features, "
                  f"{len(np.unique(y))} classes")
            
        except Exception as e:
            print(f"✗ {name:20s} - FAILED: {str(e)[:50]}")


def test_full_loading_pipeline():
    """Test complete loading with all options."""
    print("\n" + "="*70)
    print("Testing Full Loading Pipeline")
    print("="*70)
    
    name = 'heart'
    
    try:
        print(f"\nTesting {name} with various options...")
        
        # Test 1: Basic load
        data1 = load_benchmark_dataset(name, test_size=0.2)
        print(f"  ✓ Basic load: {len(data1['X_train'])} train, {len(data1['X_test'])} test")
        
        # Test 2: With standardization
        data2 = load_benchmark_dataset(name, test_size=0.2, standardize=True)
        print(f"  ✓ Standardization: mean={data2['X_train'].mean():.4f}, "
              f"std={data2['X_train'].std():.4f}")
        
        # Test 3: With stratification
        data3 = load_benchmark_dataset(name, test_size=0.2, stratify=True)
        train_dist = np.bincount(data3['y_train'])
        test_dist = np.bincount(data3['y_test'])
        print(f"  ✓ Stratification: train={train_dist}, test={test_dist}")
        
        # Test 4: Check metadata
        meta = data3['metadata']
        print(f"  ✓ Metadata: {meta['n_samples']} samples, "
              f"{meta['n_features']} features, {meta['n_classes']} classes")
        
        print(f"\n✓ {name} passed all tests!")
        
    except Exception as e:
        print(f"\n✗ {name} failed: {e}")
        import traceback
        traceback.print_exc()


def test_with_ga_training():
    """Test integration with GA training."""
    print("\n" + "="*70)
    print("Testing Integration with GA Training")
    print("="*70)
    
    try:
        # Load dataset
        data = load_benchmark_dataset('heart', test_size=0.2, standardize=True)
        X_train = data['X_train']
        y_train = data['y_train']
        X_test = data['X_test']
        y_test = data['y_test']
        
        print(f"✓ Dataset loaded: {len(X_train)} train, {len(X_test)} test")
        
        # Quick GA training (small population for speed)
        from ga_trees.ga.engine import GAEngine, GAConfig, TreeInitializer, Mutation
        from ga_trees.fitness.calculator import FitnessCalculator, TreePredictor
        
        n_features = data['metadata']['n_features']
        n_classes = data['metadata']['n_classes']
        feature_ranges = {i: (X_train[:, i].min(), X_train[:, i].max()) 
                         for i in range(n_features)}
        
        ga_config = GAConfig(population_size=10, n_generations=5)
        initializer = TreeInitializer(n_features=n_features, n_classes=n_classes,
                                     max_depth=4, min_samples_split=5, min_samples_leaf=2)
        fitness_calc = FitnessCalculator()
        mutation = Mutation(n_features=n_features, feature_ranges=feature_ranges)
        
        print("  Training GA (10 pop, 5 gen)...", end=' ', flush=True)
        ga_engine = GAEngine(ga_config, initializer, 
                            fitness_calc.calculate_fitness, mutation)
        best_tree = ga_engine.evolve(X_train, y_train, verbose=False)
        
        # Evaluate
        predictor = TreePredictor()
        y_pred = predictor.predict(best_tree, X_test)
        
        from sklearn.metrics import accuracy_score
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"Done!")
        print(f"✓ Training complete: Accuracy={accuracy:.4f}, "
              f"Nodes={best_tree.get_num_nodes()}")
        
        print("\n✓ GA integration works!")
        
    except Exception as e:
        print(f"\n✗ GA integration failed: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Run all tests."""
    print("\n" + "="*70)
    print("Dataset Loader Fix - Verification Test")
    print("="*70)
    print("\nThis will test:")
    print("  1. Sklearn datasets (should work)")
    print("  2. OpenML datasets (previously broken, now fixed)")
    print("  3. Full loading pipeline with all options")
    print("  4. Integration with GA training")
    
    test_sklearn_datasets()
    test_openml_datasets()
    test_full_loading_pipeline()
    test_with_ga_training()
    
    print("\n" + "="*70)
    print("Test Complete!")
    print("="*70)
    print("\nIf all tests passed (✓), you can now:")
    print("  • Use train.py with any dataset")
    print("  • Use experiment.py with OpenML datasets")
    print("  • Load custom CSV/Excel files")
    print("\nNext steps:")
    print("  python scripts/train.py --config configs/custom.yaml --dataset heart")
    print("  python scripts/experiment_enhanced.py --config configs/custom.yaml --datasets iris heart")
    print("\n" + "="*70 + "\n")


if __name__ == '__main__':
    main()