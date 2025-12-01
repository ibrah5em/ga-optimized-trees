"""
Dataset Loader Integration Helper

This module provides seamless integration between the enhanced dataset loader
and existing GA training scripts (train.py, experiment.py, etc.)

Usage in scripts:
    from dataset_integration import load_any_dataset
    X, y = load_any_dataset('titanic')
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import numpy as np
from typing import Tuple
from sklearn.datasets import load_iris, load_wine, load_breast_cancer


def load_any_dataset(name: str, standardize: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load dataset from any source (sklearn, OpenML, or custom file).
    
    This function provides a unified interface compatible with existing scripts.
    
    Args:
        name: Dataset name or path to file
        standardize: Whether to standardize features
        
    Returns:
        Tuple of (X, y) - full dataset without train/test split
        
    Examples:
        # Load sklearn dataset
        X, y = load_any_dataset('iris')
        
        # Load OpenML dataset
        X, y = load_any_dataset('titanic')
        
        # Load custom CSV
        X, y = load_any_dataset('data/my_data.csv')
    """
    # First try built-in sklearn datasets (fast path)
    if name in ['iris', 'wine', 'breast_cancer']:
        if name == 'iris':
            X, y = load_iris(return_X_y=True)
        elif name == 'wine':
            X, y = load_wine(return_X_y=True)
        elif name == 'breast_cancer':
            X, y = load_breast_cancer(return_X_y=True)
        
        if standardize:
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            X = scaler.fit_transform(X)
        
        return X, y
    
    # Otherwise use the enhanced loader
    try:
        from ga_trees.data.dataset_loader import DatasetLoader
        
        loader = DatasetLoader()
        
        # Load with minimal test split, then combine
        data = loader.load_dataset(
            name, 
            test_size=0.2,  # Use reasonable split
            standardize=standardize,
            stratify=True
        )
        
        # Combine train and test to return full dataset
        X = np.vstack([data['X_train'], data['X_test']])
        y = np.hstack([data['y_train'], data['y_test']])
        
        return X, y
        
    except Exception as e:
        raise ValueError(f"Failed to load dataset '{name}': {e}")


def get_dataset_info(name: str) -> dict:
    """
    Get information about a dataset without loading it.
    
    Args:
        name: Dataset name
        
    Returns:
        Dictionary with dataset metadata
    """
    from ga_trees.data.dataset_loader import DatasetLoader
    return DatasetLoader.get_dataset_info(name)


def list_all_datasets() -> dict:
    """
    List all available datasets.
    
    Returns:
        Dictionary with dataset categories
    """
    from ga_trees.data.dataset_loader import DatasetLoader
    return DatasetLoader.list_available_datasets()


# Update the load_dataset function in experiment.py to use this
def load_dataset_for_experiment(name: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load dataset for experiment.py script.
    
    This is a drop-in replacement for the existing load_dataset function.
    """
    return load_any_dataset(name, standardize=False)


if __name__ == '__main__':
    # Test the integration
    print("Testing Dataset Integration...")
    print("="*70)
    
    # Test 1: sklearn dataset
    print("\n1. Loading sklearn dataset (iris)...")
    X, y = load_any_dataset('iris')
    print(f"   ✓ Loaded: {X.shape[0]} samples, {X.shape[1]} features")
    
    # Test 2: OpenML dataset
    print("\n2. Loading OpenML dataset (heart)...")
    try:
        X, y = load_any_dataset('heart')
        print(f"   ✓ Loaded: {X.shape[0]} samples, {X.shape[1]} features")
    except Exception as e:
        print(f"   ✗ Failed: {e}")
    
    # Test 3: List available
    print("\n3. Listing available datasets...")
    available = list_all_datasets()
    print(f"   ✓ {len(available['sklearn'])} sklearn datasets")
    print(f"   ✓ {len(available['openml'])} OpenML datasets")
    
    print("\n" + "="*70)
    print("Integration test complete!")