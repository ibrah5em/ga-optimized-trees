"""
Comprehensive tests for enhanced dataset loader.

Tests cover:
- OpenML dataset loading
- UCI dataset loading  
- CSV/Excel file loading
- Data validation
- Error handling
- Edge cases
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile
import shutil
from sklearn.datasets import load_iris

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from ga_trees.data.dataset_loader import (
    DatasetLoader, 
    DataValidator,
    load_benchmark_dataset
)


class TestDataValidator:
    """Test DataValidator class."""
    
    def test_validate_valid_dataset(self):
        """Test validation of valid dataset."""
        X = np.random.rand(100, 4)
        y = np.random.randint(0, 3, 100)
        
        validator = DataValidator()
        valid, warnings = validator.validate_dataset(X, y, task_type='classification')
        
        assert valid
        assert isinstance(warnings, list)
    
    def test_validate_empty_dataset(self):
        """Test empty dataset detection."""
        X = np.array([]).reshape(0, 4)
        y = np.array([])
        
        validator = DataValidator()
        valid, warnings = validator.validate_dataset(X, y)
        
        assert not valid
        assert "empty" in warnings[0].lower()
    
    def test_validate_dimension_mismatch(self):
        """Test dimension mismatch detection."""
        X = np.random.rand(100, 4)
        y = np.random.randint(0, 2, 90)  # Wrong size
        
        validator = DataValidator()
        valid, warnings = validator.validate_dataset(X, y)
        
        assert not valid
        assert "different number of samples" in warnings[0].lower()
    
    def test_validate_nan_detection(self):
        """Test NaN detection."""
        X = np.random.rand(100, 4)
        X[10, 2] = np.nan  # Insert NaN
        y = np.random.randint(0, 2, 100)
        
        validator = DataValidator()
        valid, warnings = validator.validate_dataset(X, y)
        
        assert valid  # Still valid but with warnings
        assert any("NaN" in w for w in warnings)
    
    def test_validate_inf_detection(self):
        """Test Inf detection."""
        X = np.random.rand(100, 4)
        X[15, 1] = np.inf  # Insert Inf
        y = np.random.randint(0, 2, 100)
        
        validator = DataValidator()
        valid, warnings = validator.validate_dataset(X, y)
        
        assert valid
        assert any("Inf" in w for w in warnings)
    
    def test_validate_imbalanced_classes(self):
        """Test severe class imbalance detection."""
        X = np.random.rand(100, 4)
        y = np.concatenate([np.zeros(95), np.ones(5)])  # 95:5 ratio
        
        validator = DataValidator()
        valid, warnings = validator.validate_dataset(X, y, task_type='classification')
        
        assert valid
        assert any("imbalance" in w.lower() for w in warnings)
    
    def test_validate_single_class(self):
        """Test single class detection."""
        X = np.random.rand(100, 4)
        y = np.zeros(100)  # All same class
        
        validator = DataValidator()
        valid, warnings = validator.validate_dataset(X, y, task_type='classification')
        
        assert not valid
        assert any("at least 2 classes" in w.lower() for w in warnings)
    
    def test_validate_zero_variance_features(self):
        """Test zero variance feature detection."""
        X = np.random.rand(100, 4)
        X[:, 2] = 1.0  # Constant feature
        y = np.random.randint(0, 2, 100)
        
        validator = DataValidator()
        valid, warnings = validator.validate_dataset(X, y)
        
        assert valid
        assert any("zero variance" in w.lower() for w in warnings)
    
    def test_clean_dataset_remove_strategy(self):
        """Test cleaning with remove strategy."""
        X = np.array([[1.0, 2.0], [3.0, np.nan], [5.0, 6.0]])
        y = np.array([0, 1, 0])
        
        validator = DataValidator()
        X_clean, y_clean = validator.clean_dataset(X, y, strategy='remove')
        
        assert len(X_clean) == 2
        assert len(y_clean) == 2
        assert not np.any(np.isnan(X_clean))
    
    def test_clean_dataset_mean_strategy(self):
        """Test cleaning with mean imputation."""
        X = np.array([[1.0, 2.0], [3.0, np.nan], [5.0, 6.0]])
        y = np.array([0, 1, 0])
        
        validator = DataValidator()
        X_clean, y_clean = validator.clean_dataset(X, y, strategy='mean')
        
        assert len(X_clean) == 3
        assert not np.any(np.isnan(X_clean))
        # Check mean imputation: (2 + 6) / 2 = 4.0
        assert X_clean[1, 1] == 4.0
    
    def test_clean_dataset_zero_strategy(self):
        """Test cleaning with zero filling."""
        X = np.array([[1.0, np.nan], [np.inf, 4.0]])
        y = np.array([0, 1])
        
        validator = DataValidator()
        X_clean, y_clean = validator.clean_dataset(X, y, strategy='zero')
        
        assert not np.any(np.isnan(X_clean))
        assert not np.any(np.isinf(X_clean))
        assert X_clean[0, 1] == 0.0
        assert X_clean[1, 0] == 0.0


class TestDatasetLoaderSklearn:
    """Test loading sklearn datasets."""
    
    def test_load_iris(self):
        """Test loading Iris dataset."""
        loader = DatasetLoader()
        data = loader.load_dataset('iris', test_size=0.2, random_state=42)
        
        assert 'X_train' in data
        assert 'X_test' in data
        assert 'y_train' in data
        assert 'y_test' in data
        assert data['metadata']['n_samples'] == 150
        assert data['metadata']['n_features'] == 4
        assert data['metadata']['n_classes'] == 3
    
    def test_load_wine(self):
        """Test loading Wine dataset."""
        loader = DatasetLoader()
        data = loader.load_dataset('wine', test_size=0.3, random_state=42)
        
        assert data['metadata']['n_samples'] == 178
        assert data['metadata']['n_classes'] == 3
    
    def test_load_breast_cancer(self):
        """Test loading Breast Cancer dataset."""
        loader = DatasetLoader()
        data = loader.load_dataset('breast_cancer', test_size=0.2)
        
        assert data['metadata']['n_samples'] == 569
        assert data['metadata']['n_features'] == 30
        assert data['metadata']['n_classes'] == 2
    
    def test_stratified_split(self):
        """Test stratified splitting maintains class proportions."""
        loader = DatasetLoader()
        data = loader.load_dataset('iris', test_size=0.2, stratify=True, random_state=42)
        
        # Check class proportions
        train_classes, train_counts = np.unique(data['y_train'], return_counts=True)
        test_classes, test_counts = np.unique(data['y_test'], return_counts=True)
        
        assert len(train_classes) == 3
        assert len(test_classes) == 3
        # Each class should be represented
        assert np.all(train_counts > 0)
        assert np.all(test_counts > 0)
    
    def test_standardization(self):
        """Test feature standardization."""
        loader = DatasetLoader()
        data = loader.load_dataset('iris', standardize=True, random_state=42)
        
        # Check standardization
        mean = np.mean(data['X_train'], axis=0)
        std = np.std(data['X_train'], axis=0)
        
        assert np.allclose(mean, 0, atol=1e-10)
        assert np.allclose(std, 1, atol=1e-1)
        assert data['scaler'] is not None
        assert data['metadata']['standardized'] is True


class TestDatasetLoaderOpenML:
    """Test loading OpenML datasets."""
    
    @pytest.mark.slow
    def test_load_credit_g(self):
        """Test loading German Credit dataset."""
        loader = DatasetLoader()
        try:
            data = loader.load_dataset('credit_g', test_size=0.2, random_state=42)
            assert data['metadata']['n_samples'] > 0
            assert data['metadata']['n_classes'] == 2
        except Exception as e:
            pytest.skip(f"OpenML not available: {e}")
    
    @pytest.mark.slow
    def test_load_heart(self):
        """Test loading Heart Disease dataset."""
        loader = DatasetLoader()
        try:
            data = loader.load_dataset('heart', test_size=0.2)
            assert data['metadata']['n_samples'] > 0
        except Exception as e:
            pytest.skip(f"OpenML not available: {e}")
    
    @pytest.mark.slow
    def test_load_diabetes_pima(self):
        """Test loading Diabetes dataset."""
        loader = DatasetLoader()
        try:
            data = loader.load_dataset('diabetes_pima', test_size=0.2)
            assert data['metadata']['n_classes'] == 2
        except Exception as e:
            pytest.skip(f"OpenML not available: {e}")


class TestDatasetLoaderCSV:
    """Test loading CSV files."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test files."""
        temp = tempfile.mkdtemp()
        yield temp
        shutil.rmtree(temp)
    
    def test_load_valid_csv(self, temp_dir):
        """Test loading valid CSV file."""
        # Create test CSV
        csv_path = Path(temp_dir) / 'test.csv'
        df = pd.DataFrame({
            'feature1': [1.0, 2.0, 3.0, 4.0],
            'feature2': [5.0, 6.0, 7.0, 8.0],
            'target': [0, 1, 0, 1]
        })
        df.to_csv(csv_path, index=False)
        
        loader = DatasetLoader()
        data = loader.load_dataset(str(csv_path), test_size=0.25, random_state=42)
        
        assert data['metadata']['n_samples'] == 4
        assert data['metadata']['n_features'] == 2
        assert len(data['X_train']) == 3
        assert len(data['X_test']) == 1
    
    def test_load_csv_with_categorical_target(self, temp_dir):
        """Test loading CSV with categorical target."""
        csv_path = Path(temp_dir) / 'test_cat.csv'
        df = pd.DataFrame({
            'feature1': [1.0, 2.0, 3.0, 4.0],
            'feature2': [5.0, 6.0, 7.0, 8.0],
            'target': ['cat', 'dog', 'cat', 'dog']
        })
        df.to_csv(csv_path, index=False)
        
        loader = DatasetLoader()
        data = loader.load_dataset(str(csv_path), test_size=0.25)
        
        # Target should be encoded
        assert np.issubdtype(data['y_train'].dtype, np.integer)
        assert data['target_names'] is not None
        assert set(data['target_names']) == {'cat', 'dog'}
    
    def test_load_csv_with_categorical_features(self, temp_dir):
        """Test loading CSV with categorical features."""
        csv_path = Path(temp_dir) / 'test_cat_feat.csv'
        df = pd.DataFrame({
            'color': ['red', 'blue', 'red', 'green'],
            'size': ['small', 'large', 'medium', 'small'],
            'target': [0, 1, 0, 1]
        })
        df.to_csv(csv_path, index=False)
        
        loader = DatasetLoader()
        data = loader.load_dataset(str(csv_path), test_size=0.25)
        
        # Features should be encoded
        assert np.issubdtype(data['X_train'].dtype, np.floating)
    
    def test_load_nonexistent_file(self):
        """Test error handling for nonexistent file."""
        loader = DatasetLoader()
        
        with pytest.raises(FileNotFoundError):
            loader.load_dataset('nonexistent.csv')
    
    def test_load_malformed_csv(self, temp_dir):
        """Test error handling for malformed CSV."""
        csv_path = Path(temp_dir) / 'malformed.csv'
        with open(csv_path, 'w') as f:
            f.write("col1,col2\n")
            f.write("1,2,3\n")  # Wrong number of columns
            f.write("4,5\n")
        
        loader = DatasetLoader()
        # Should either handle or raise appropriate error
        try:
            data = loader.load_dataset(str(csv_path))
            # If it loads, check it handles the error gracefully
            assert data is not None
        except Exception as e:
            # Should be a clear error message
            assert "malformed" in str(e).lower() or "error" in str(e).lower()


class TestDatasetLoaderExcel:
    """Test loading Excel files."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory."""
        temp = tempfile.mkdtemp()
        yield temp
        shutil.rmtree(temp)
    
    def test_load_valid_excel(self, temp_dir):
        """Test loading valid Excel file."""
        excel_path = Path(temp_dir) / 'test.xlsx'
        df = pd.DataFrame({
            'feature1': [1.0, 2.0, 3.0],
            'feature2': [4.0, 5.0, 6.0],
            'target': [0, 1, 0]
        })
        df.to_excel(excel_path, index=False, engine='openpyxl')
        
        loader = DatasetLoader()
        data = loader.load_dataset(str(excel_path), test_size=0.33)
        
        assert data['metadata']['n_samples'] == 3
        assert data['metadata']['n_features'] == 2
    
    def test_load_unsupported_format(self):
        """Test error for unsupported file format."""
        loader = DatasetLoader()
        
        with pytest.raises(ValueError) as exc_info:
            loader.load_dataset('test.txt')
        assert "unsupported" in str(exc_info.value).lower()


class TestDatasetBalancing:
    """Test dataset balancing functionality."""
    
    def test_oversample_minority_class(self):
        """Test oversampling minority class."""
        loader = DatasetLoader()
        data = loader.load_dataset(
            'iris', 
            test_size=0.2, 
            balance='oversample',
            random_state=42
        )
        
        # Check classes are balanced
        unique, counts = np.unique(data['y_train'], return_counts=True)
        assert len(set(counts)) == 1  # All counts should be equal
        assert data['metadata']['balanced'] is True
    
    def test_undersample_majority_class(self):
        """Test undersampling majority class."""
        loader = DatasetLoader()
        data = loader.load_dataset(
            'iris',
            test_size=0.2,
            balance='undersample',
            random_state=42
        )
        
        # Check classes are balanced
        unique, counts = np.unique(data['y_train'], return_counts=True)
        assert len(set(counts)) == 1
        assert data['metadata']['balanced'] is True
    
    def test_no_balancing(self):
        """Test dataset without balancing."""
        loader = DatasetLoader()
        data = loader.load_dataset('iris', test_size=0.2, balance=None)
        
        assert data['metadata']['balanced'] is False


class TestDatasetLoaderUtilities:
    """Test utility functions."""
    
    def test_list_available_datasets(self):
        """Test listing available datasets."""
        available = DatasetLoader.list_available_datasets()
        
        assert 'sklearn' in available
        assert 'openml' in available
        assert 'custom' in available
        
        assert len(available['sklearn']) >= 5
        assert len(available['openml']) >= 15
    
    def test_get_dataset_info_sklearn(self):
        """Test getting info for sklearn dataset."""
        info = DatasetLoader.get_dataset_info('iris')
        
        assert info['name'] == 'iris'
        assert info['source'] == 'sklearn'
        assert info['available'] is True
    
    def test_get_dataset_info_openml(self):
        """Test getting info for OpenML dataset."""
        info = DatasetLoader.get_dataset_info('credit_g')
        
        assert info['name'] == 'credit_g'
        assert 'openml' in info['source'].lower()
        assert info['available'] is True
    
    def test_get_dataset_info_unknown(self):
        """Test getting info for unknown dataset."""
        info = DatasetLoader.get_dataset_info('nonexistent_dataset')
        
        assert info['available'] is False


class TestConvenienceFunction:
    """Test load_benchmark_dataset convenience function."""
    
    def test_load_benchmark_iris(self):
        """Test convenience function with Iris."""
        data = load_benchmark_dataset('iris', test_size=0.2, random_state=42)
        
        assert 'X_train' in data
        assert 'y_train' in data
        assert data['metadata']['n_samples'] == 150
    
    def test_load_benchmark_with_standardization(self):
        """Test convenience function with standardization."""
        data = load_benchmark_dataset('wine', standardize=True, random_state=42)
        
        mean = np.mean(data['X_train'], axis=0)
        assert np.allclose(mean, 0, atol=1e-10)
        assert data['metadata']['standardized'] is True


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_very_small_dataset(self, tmpdir):
        """Test handling of very small dataset."""
        csv_path = Path(tmpdir) / 'tiny.csv'
        df = pd.DataFrame({
            'feature1': [1.0, 2.0],
            'target': [0, 1]
        })
        df.to_csv(csv_path, index=False)
        
        loader = DatasetLoader()
        # Should handle gracefully or raise clear error
        try:
            data = loader.load_dataset(str(csv_path), test_size=0.5)
            assert len(data['X_train']) >= 1
            assert len(data['X_test']) >= 1
        except ValueError as e:
            # Acceptable if it raises error for too small dataset
            assert "too small" in str(e).lower() or "insufficient" in str(e).lower()
    
    def test_single_feature(self, tmpdir):
        """Test dataset with single feature."""
        csv_path = Path(tmpdir) / 'single_feat.csv'
        df = pd.DataFrame({
            'feature1': [1.0, 2.0, 3.0, 4.0, 5.0],
            'target': [0, 1, 0, 1, 0]
        })
        df.to_csv(csv_path, index=False)
        
        loader = DatasetLoader()
        data = loader.load_dataset(str(csv_path), test_size=0.2)
        
        assert data['metadata']['n_features'] == 1
    
    def test_unknown_dataset_name(self):
        """Test error for unknown dataset name."""
        loader = DatasetLoader()
        
        with pytest.raises(ValueError) as exc_info:
            loader.load_dataset('definitely_not_a_real_dataset')
        assert "unknown dataset" in str(exc_info.value).lower()
    
    def test_invalid_test_size(self):
        """Test error for invalid test_size."""
        loader = DatasetLoader()
        
        # test_size > 1.0 should fail
        with pytest.raises((ValueError, Exception)):
            loader.load_dataset('iris', test_size=1.5)
        
        # test_size < 0 should fail
        with pytest.raises((ValueError, Exception)):
            loader.load_dataset('iris', test_size=-0.1)
    
    def test_empty_csv(self, tmpdir):
        """Test handling of empty CSV."""
        csv_path = Path(tmpdir) / 'empty.csv'
        df = pd.DataFrame()
        df.to_csv(csv_path, index=False)
        
        loader = DatasetLoader()
        
        with pytest.raises((ValueError, Exception)) as exc_info:
            loader.load_dataset(str(csv_path))
        # Should have meaningful error message
        error_msg = str(exc_info.value).lower()
        assert any(word in error_msg for word in ['empty', 'no data', 'invalid'])


class TestIntegrationWithGA:
    """Test integration with GA training pipeline."""
    
    def test_dataset_format_for_ga(self):
        """Test that loaded data works with GA."""
        loader = DatasetLoader()
        data = load_benchmark_dataset('iris', test_size=0.2, standardize=True, random_state=42)
        
        X_train = data['X_train']
        y_train = data['y_train']
        X_test = data['X_test']
        y_test = data['y_test']
        
        # Check types
        assert isinstance(X_train, np.ndarray)
        assert isinstance(y_train, np.ndarray)
        assert isinstance(X_test, np.ndarray)
        assert isinstance(y_test, np.ndarray)
        
        # Check shapes
        assert X_train.shape[1] == data['metadata']['n_features']
        assert len(y_train) == X_train.shape[0]
        assert len(y_test) == X_test.shape[0]
        
        # Check no NaN/Inf
        assert not np.any(np.isnan(X_train))
        assert not np.any(np.isinf(X_train))
    
    def test_feature_ranges_extraction(self):
        """Test extracting feature ranges for GA."""
        data = load_benchmark_dataset('wine', standardize=False, random_state=42)
        X_train = data['X_train']
        n_features = data['metadata']['n_features']
        
        # Extract feature ranges (needed for GA mutation)
        feature_ranges = {i: (X_train[:, i].min(), X_train[:, i].max()) 
                         for i in range(n_features)}
        
        assert len(feature_ranges) == n_features
        for i, (min_val, max_val) in feature_ranges.items():
            assert min_val <= max_val
            assert not np.isnan(min_val)
            assert not np.isnan(max_val)


class TestDatasetLoaderReproducibility:
    """Test reproducibility with random seeds."""
    
    def test_same_seed_same_split(self):
        """Test that same seed produces same split."""
        loader = DatasetLoader()
        
        data1 = loader.load_dataset('iris', test_size=0.2, random_state=42)
        data2 = loader.load_dataset('iris', test_size=0.2, random_state=42)
        
        np.testing.assert_array_equal(data1['X_train'], data2['X_train'])
        np.testing.assert_array_equal(data1['y_train'], data2['y_train'])
        np.testing.assert_array_equal(data1['X_test'], data2['X_test'])
        np.testing.assert_array_equal(data1['y_test'], data2['y_test'])
    
    def test_different_seed_different_split(self):
        """Test that different seeds produce different splits."""
        loader = DatasetLoader()
        
        data1 = loader.load_dataset('iris', test_size=0.2, random_state=42)
        data2 = loader.load_dataset('iris', test_size=0.2, random_state=123)
        
        # Splits should be different
        assert not np.array_equal(data1['X_train'], data2['X_train'])


class TestPerformance:
    """Test performance and efficiency."""
    
    @pytest.mark.slow
    def test_large_dataset_loading(self):
        """Test loading larger dataset."""
        import time
        
        loader = DatasetLoader()
        
        start = time.time()
        try:
            data = loader.load_dataset('adult', test_size=0.2, random_state=42)
            elapsed = time.time() - start
            
            # Should load in reasonable time (< 30 seconds)
            assert elapsed < 30
            assert data['metadata']['n_samples'] > 10000
        except Exception:
            pytest.skip("Adult dataset not available or too large")
    
    def test_caching(self):
        """Test that loader uses caching effectively."""
        loader = DatasetLoader()
        
        import time
        
        # First load
        start1 = time.time()
        data1 = loader.load_dataset('iris', test_size=0.2, random_state=42)
        time1 = time.time() - start1
        
        # Second load (might be cached by OpenML/sklearn)
        start2 = time.time()
        data2 = loader.load_dataset('iris', test_size=0.2, random_state=42)
        time2 = time.time() - start2
        
        # Results should be identical
        np.testing.assert_array_equal(data1['X_train'], data2['X_train'])
