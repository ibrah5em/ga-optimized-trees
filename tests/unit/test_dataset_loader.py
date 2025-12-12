"""
Pytest-compatible tests for Dataset Loader

FILE PATH: /home/yousef/ga-optimized-trees/tests/unit/test_dataset_loader.py
ACTION: REPLACE or CREATE

Usage:
    # Run all dataset loader tests
    pytest tests/unit/test_dataset_loader.py -v

    # Run with coverage
    pytest tests/unit/test_dataset_loader.py -v --cov=src/ga_trees/data

    # Run specific test
    pytest tests/unit/test_dataset_loader.py::TestSklearnDatasets::test_load_iris -v

    # Run only fast tests (skip slow OpenML tests)
    pytest tests/unit/test_dataset_loader.py -v -m "not slow"
"""

import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# FIXED: we now Use correct import path
from src.ga_trees.data.dataset_loader import DatasetLoader, DataValidator, load_benchmark_dataset

# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def loader():
    """Create DatasetLoader instance."""
    return DatasetLoader()


@pytest.fixture
def validator():
    """Create DataValidator instance."""
    return DataValidator()


@pytest.fixture
def temp_dir():
    """Create temporary directory for test files."""
    temp = tempfile.mkdtemp()
    yield temp
    shutil.rmtree(temp)


@pytest.fixture
def sample_csv_path(temp_dir):
    """Create sample CSV file."""
    csv_path = Path(temp_dir) / "test.csv"
    df = pd.DataFrame(
        {
            "feature1": np.random.rand(20),
            "feature2": np.random.rand(20),
            "target": np.random.randint(0, 2, 20),
        }
    )
    df.to_csv(csv_path, index=False)
    return str(csv_path)


# ============================================================================
# SKLEARN DATASET TESTS
# ============================================================================


class TestSklearnDatasets:
    """Tests for sklearn built-in datasets."""

    def test_load_iris(self, loader):
        """Test loading Iris dataset."""
        data = loader.load_dataset("iris", test_size=0.2, random_state=42)

        assert data is not None
        assert "X_train" in data
        assert "y_train" in data
        assert "X_test" in data
        assert "y_test" in data
        assert data["metadata"]["n_samples"] == 150
        assert data["metadata"]["n_features"] == 4
        assert data["metadata"]["n_classes"] == 3

    def test_load_wine(self, loader):
        """Test loading Wine dataset."""
        data = loader.load_dataset("wine", test_size=0.3, random_state=42)

        assert data["metadata"]["n_samples"] == 178
        assert data["metadata"]["n_classes"] == 3

    def test_load_breast_cancer(self, loader):
        """Test loading Breast Cancer dataset."""
        data = loader.load_dataset("breast_cancer", test_size=0.2, random_state=42)

        assert data["metadata"]["n_samples"] == 569
        assert data["metadata"]["n_features"] == 30
        assert data["metadata"]["n_classes"] == 2


# ============================================================================
# OPENML DATASET TESTS
# ============================================================================


@pytest.mark.slow
class TestOpenMLDatasets:
    """Tests for OpenML datasets (marked as slow)."""

    def test_load_heart(self, loader):
        """Test loading Heart Disease dataset from OpenML."""
        data = loader.load_dataset("heart", test_size=0.2, random_state=42)

        assert data is not None
        assert data["metadata"]["n_samples"] > 0
        assert data["metadata"]["n_classes"] == 2

        # Check data types
        assert np.issubdtype(data["X_train"].dtype, np.floating)
        assert np.issubdtype(data["y_train"].dtype, np.integer)

        # Check for NaN
        assert not np.any(np.isnan(data["X_train"]))
        assert not np.any(np.isnan(data["y_train"]))

    def test_load_titanic(self, loader):
        """Test loading Titanic dataset from OpenML."""
        data = loader.load_dataset("titanic", test_size=0.2, random_state=42)

        assert data["metadata"]["n_samples"] > 1000
        assert data["metadata"]["n_classes"] == 2

        # Verify no categorical data leaked through
        assert np.issubdtype(data["X_train"].dtype, np.floating)

    def test_load_credit_g(self, loader):
        """Test loading German Credit dataset."""
        data = loader.load_dataset("credit_g", test_size=0.2, random_state=42)

        assert data["metadata"]["n_samples"] == 1000
        assert data["metadata"]["n_classes"] == 2


# ============================================================================
# CSV/EXCEL FILE TESTS
# ============================================================================


class TestCSVLoading:
    """Tests for CSV file loading."""

    def test_load_basic_csv(self, loader, temp_dir):
        """Test loading basic CSV file."""
        csv_path = Path(temp_dir) / "test.csv"
        df = pd.DataFrame(
            {
                "feature1": np.random.rand(20),
                "feature2": np.random.rand(20),
                "target": np.random.randint(0, 2, 20),
            }
        )
        df.to_csv(csv_path, index=False)

        data = loader.load_dataset(str(csv_path), test_size=0.2, random_state=42)

        assert data["metadata"]["n_samples"] == 20
        assert data["metadata"]["n_features"] == 2
        assert len(data["X_train"]) == 16
        assert len(data["X_test"]) == 4

    def test_load_csv_categorical_target(self, loader, temp_dir):
        """Test CSV with categorical target."""
        csv_path = Path(temp_dir) / "test_cat.csv"
        df = pd.DataFrame(
            {
                "feature1": np.random.rand(20),
                "feature2": np.random.rand(20),
                "target": ["cat", "dog"] * 10,
            }
        )
        df.to_csv(csv_path, index=False)

        data = loader.load_dataset(str(csv_path), test_size=0.2, random_state=42)

        # Target should be encoded to integers
        assert np.issubdtype(data["y_train"].dtype, np.integer)
        assert data["target_names"] is not None
        assert set(data["target_names"]) == {"cat", "dog"}

    def test_load_csv_with_nan(self, loader, temp_dir):
        """Test CSV with missing values."""
        csv_path = Path(temp_dir) / "test_nan.csv"
        data_values = np.random.rand(15, 2)
        data_values[3, 0] = np.nan
        data_values[7, 1] = np.nan

        df = pd.DataFrame(
            {
                "feature1": data_values[:, 0],
                "feature2": data_values[:, 1],
                "target": np.random.randint(0, 2, 15),
            }
        )
        df.to_csv(csv_path, index=False)

        data = loader.load_dataset(str(csv_path), test_size=0.2, random_state=42)

        # Should handle NaN (clean with median by default)
        assert not np.any(np.isnan(data["X_train"]))
        assert not np.any(np.isnan(data["X_test"]))

    def test_load_csv_very_small(self, loader, temp_dir):
        """Test CSV with small dataset (edge case)."""
        csv_path = Path(temp_dir) / "tiny.csv"
        df = pd.DataFrame(
            {
                "feature1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
                "feature2": [5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0],
                "target": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
            }
        )
        df.to_csv(csv_path, index=False)

        # Should work with small dataset
        data = loader.load_dataset(str(csv_path), test_size=0.2, random_state=42)

        assert data is not None
        assert len(data["X_train"]) == 8
        assert len(data["X_test"]) == 2

    def test_load_csv_categorical_features(self, loader, temp_dir):
        """Test CSV with categorical features."""
        csv_path = Path(temp_dir) / "test_cat_feat.csv"
        df = pd.DataFrame(
            {
                "color": ["red", "blue", "red", "green"] * 5,
                "size": ["small", "large", "medium", "small"] * 5,
                "target": [0, 1, 0, 1] * 5,
            }
        )
        df.to_csv(csv_path, index=False)

        data = loader.load_dataset(str(csv_path), test_size=0.25, random_state=42)

        # Features should be encoded
        assert np.issubdtype(data["X_train"].dtype, np.floating)


# ============================================================================
# DATA VALIDATION TESTS
# ============================================================================


class TestDataValidation:
    """Tests for DataValidator class."""

    def test_validate_valid_dataset(self, validator):
        """Test validation of valid dataset."""
        X = np.random.rand(100, 4)
        y = np.random.randint(0, 3, 100)

        valid, warnings = validator.validate_dataset(X, y, task_type="classification")

        assert valid
        assert isinstance(warnings, list)

    def test_validate_empty_dataset(self, validator):
        """Test empty dataset detection."""
        X = np.array([]).reshape(0, 4)
        y = np.array([])

        valid, warnings = validator.validate_dataset(X, y)

        assert not valid
        assert any("empty" in w.lower() for w in warnings)

    def test_validate_dimension_mismatch(self, validator):
        """Test dimension mismatch detection."""
        X = np.random.rand(100, 4)
        y = np.random.randint(0, 2, 90)  # Wrong size

        valid, warnings = validator.validate_dataset(X, y)

        assert not valid
        assert any("different" in w.lower() for w in warnings)

    def test_validate_nan_detection(self, validator):
        """Test NaN detection."""
        X = np.random.rand(100, 4)
        X[10, 2] = np.nan
        y = np.random.randint(0, 2, 100)

        valid, warnings = validator.validate_dataset(X, y)

        assert valid  # Valid but with warnings
        assert any("nan" in w.lower() for w in warnings)

    def test_validate_inf_detection(self, validator):
        """Test Inf detection."""
        X = np.random.rand(100, 4)
        X[15, 1] = np.inf
        y = np.random.randint(0, 2, 100)

        valid, warnings = validator.validate_dataset(X, y)

        assert valid
        assert any("inf" in w.lower() for w in warnings)

    def test_validate_class_imbalance(self, validator):
        """Test severe class imbalance detection."""
        X = np.random.rand(100, 4)
        y = np.concatenate([np.zeros(95), np.ones(5)])  # 95:5 ratio

        valid, warnings = validator.validate_dataset(X, y, task_type="classification")

        assert valid
        assert any("imbalance" in w.lower() for w in warnings)

    def test_validate_single_class(self, validator):
        """Test single class detection."""
        X = np.random.rand(100, 4)
        y = np.zeros(100)  # All same class

        valid, warnings = validator.validate_dataset(X, y, task_type="classification")

        assert not valid
        assert any("at least 2 classes" in w.lower() for w in warnings)

    def test_validate_zero_variance_features(self, validator):
        """Test zero variance feature detection."""
        X = np.random.rand(100, 4)
        X[:, 2] = 1.0  # Constant feature
        y = np.random.randint(0, 2, 100)

        valid, warnings = validator.validate_dataset(X, y)

        assert valid
        assert any("zero variance" in w.lower() for w in warnings)


# ============================================================================
# DATA CLEANING TESTS
# ============================================================================


class TestDataCleaning:
    """Tests for data cleaning functionality."""

    def test_clean_median_imputation(self, validator):
        """Test median imputation cleaning."""
        X = np.array([[1.0, 2.0], [3.0, np.nan], [5.0, 6.0]])
        y = np.array([0, 1, 0])

        X_clean, y_clean = validator.clean_dataset(X, y, strategy="median")

        assert len(X_clean) == 3
        assert not np.any(np.isnan(X_clean))
        # Median of [2.0, 6.0] = 4.0
        assert X_clean[1, 1] == 4.0

    def test_clean_mean_imputation(self, validator):
        """Test mean imputation cleaning."""
        X = np.array([[1.0, 2.0], [3.0, np.nan], [5.0, 6.0]])
        y = np.array([0, 1, 0])

        X_clean, y_clean = validator.clean_dataset(X, y, strategy="mean")

        assert len(X_clean) == 3
        assert not np.any(np.isnan(X_clean))
        # Mean of [2.0, 6.0] = 4.0
        assert X_clean[1, 1] == 4.0

    def test_clean_remove_rows(self, validator):
        """Test row removal cleaning."""
        X = np.array([[1.0, 2.0], [3.0, np.nan], [5.0, 6.0]])
        y = np.array([0, 1, 0])

        X_clean, y_clean = validator.clean_dataset(X, y, strategy="remove")

        assert len(X_clean) == 2
        assert len(y_clean) == 2
        assert not np.any(np.isnan(X_clean))

    def test_clean_zero_strategy(self, validator):
        """Test zero filling cleaning."""
        X = np.array([[1.0, np.nan], [np.inf, 4.0]])
        y = np.array([0, 1])

        X_clean, y_clean = validator.clean_dataset(X, y, strategy="zero")

        assert not np.any(np.isnan(X_clean))
        assert not np.any(np.isinf(X_clean))
        assert X_clean[0, 1] == 0.0
        assert X_clean[1, 0] == 0.0


# ============================================================================
# FEATURE TESTS
# ============================================================================


class TestFeatures:
    """Tests for feature processing."""

    def test_standardization(self, loader):
        """Test feature standardization."""
        data = loader.load_dataset("iris", standardize=True, random_state=42)

        mean = np.mean(data["X_train"], axis=0)
        std = np.std(data["X_train"], axis=0)

        assert np.allclose(mean, 0, atol=1e-10)
        assert np.allclose(std, 1, atol=0.1)
        assert data["scaler"] is not None
        assert data["metadata"]["standardized"] is True

    def test_stratified_split(self, loader):
        """Test stratified splitting."""
        data = loader.load_dataset("iris", test_size=0.2, stratify=True, random_state=42)

        train_classes, train_counts = np.unique(data["y_train"], return_counts=True)
        test_classes, test_counts = np.unique(data["y_test"], return_counts=True)

        assert len(train_classes) == 3
        assert len(test_classes) == 3
        assert np.all(train_counts > 0)
        assert np.all(test_counts > 0)

    def test_balancing_oversample(self, loader):
        """Test oversampling for class balance."""
        data = loader.load_dataset("iris", test_size=0.2, balance="oversample", random_state=42)

        unique, counts = np.unique(data["y_train"], return_counts=True)

        # Classes should be balanced
        assert len(set(counts)) == 1
        assert data["metadata"]["balanced"] is True

    def test_balancing_undersample(self, loader):
        """Test undersampling for class balance."""
        data = loader.load_dataset("iris", test_size=0.2, balance="undersample", random_state=42)

        unique, counts = np.unique(data["y_train"], return_counts=True)

        # Classes should be balanced
        assert len(set(counts)) == 1
        assert data["metadata"]["balanced"] is True


# ============================================================================
# INTEGRATION TESTS
# ============================================================================


class TestIntegration:
    """Integration tests with GA training."""

    def test_feature_ranges_extraction(self, loader):
        """Test extracting feature ranges for GA."""
        data = loader.load_dataset("wine", standardize=False, random_state=42)
        X_train = data["X_train"]
        n_features = data["metadata"]["n_features"]

        # Extract feature ranges (needed for GA mutation)
        feature_ranges = {i: (X_train[:, i].min(), X_train[:, i].max()) for i in range(n_features)}

        assert len(feature_ranges) == n_features
        for i, (min_val, max_val) in feature_ranges.items():
            assert min_val <= max_val
            assert not np.isnan(min_val)
            assert not np.isnan(max_val)

    def test_ga_compatibility(self, loader):
        """Test data format is GA-compatible."""
        data = loader.load_dataset(
            "breast_cancer", test_size=0.2, standardize=True, random_state=42
        )

        X_train = data["X_train"]
        y_train = data["y_train"]
        X_test = data["X_test"]
        y_test = data["y_test"]

        # Check types
        assert isinstance(X_train, np.ndarray)
        assert isinstance(y_train, np.ndarray)
        assert isinstance(X_test, np.ndarray)
        assert isinstance(y_test, np.ndarray)

        # Check shapes
        assert X_train.shape[1] == data["metadata"]["n_features"]
        assert len(y_train) == X_train.shape[0]
        assert len(y_test) == X_test.shape[0]

        # Check no NaN/Inf
        assert not np.any(np.isnan(X_train))
        assert not np.any(np.isinf(X_train))
        assert not np.any(np.isnan(y_train))


# ============================================================================
# ERROR HANDLING TESTS
# ============================================================================


class TestErrorHandling:
    """Tests for error handling."""

    def test_error_nonexistent_dataset(self, loader):
        """Test error for nonexistent dataset."""
        # FIXED: This will attempt OpenML fetch and fail
        # Should raise ValueError or RuntimeError
        with pytest.raises((ValueError, RuntimeError)):
            loader.load_dataset("definitely_not_a_real_dataset_12345")

    def test_error_nonexistent_dataset_clear_message(self, loader):
        """Test that error message is informative."""
        # BETTER: If you implement the try_openml parameter fix
        # This test would check for helpful error message
        try:
            loader.load_dataset("nonexistent_xyz_123")
        except (ValueError, RuntimeError) as e:
            error_msg = str(e).lower()
            # Should mention it's unknown or not found
            assert "unknown" in error_msg or "not found" in error_msg or "failed" in error_msg

    def test_error_invalid_test_size(self, loader):
        """Test error for invalid test_size."""
        with pytest.raises((ValueError, Exception)):
            loader.load_dataset("iris", test_size=1.5)

    def test_error_invalid_test_size_negative(self, loader):
        """Test error for negative test_size."""
        with pytest.raises((ValueError, Exception)):
            loader.load_dataset("iris", test_size=-0.1)

    def test_error_nonexistent_file(self, loader):
        """Test error for nonexistent file."""
        with pytest.raises(FileNotFoundError):
            loader.load_dataset("/path/to/definitely/nonexistent_file.csv")

    def test_error_nonexistent_file_with_extension(self, loader):
        """Test that files with extensions raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            loader.load_dataset("./missing_data.xlsx")

    def test_error_empty_csv(self, loader, temp_dir):
        """Test error for empty CSV."""
        csv_path = Path(temp_dir) / "empty.csv"
        # Create truly empty file
        csv_path.touch()

        with pytest.raises((ValueError, pd.errors.EmptyDataError)):
            loader.load_dataset(str(csv_path))

    def test_error_csv_with_headers_only(self, loader, temp_dir):
        """Test error for CSV with only headers (no data)."""
        csv_path = Path(temp_dir) / "headers_only.csv"
        with open(csv_path, "w") as f:
            f.write("feature1,feature2,target\n")

        with pytest.raises(ValueError, match="empty|no data"):
            loader.load_dataset(str(csv_path))

    def test_error_insufficient_columns(self, loader, temp_dir):
        """Test error for CSV with insufficient columns."""
        csv_path = Path(temp_dir) / "single_col.csv"
        df = pd.DataFrame({"only_one_column": [1, 2, 3, 4, 5]})
        df.to_csv(csv_path, index=False)

        with pytest.raises(ValueError, match="at least 2 columns"):
            loader.load_dataset(str(csv_path))

    def test_error_invalid_balance_strategy(self, loader):
        """Test error for invalid balance strategy."""
        # This might not raise error in current implementation
        # but it's good to document expected behavior
        with pytest.raises((ValueError, KeyError, AttributeError)):
            loader.load_dataset("iris", balance="invalid_strategy")

    def test_error_unsupported_file_format(self, loader, temp_dir):
        """Test error for unsupported file format."""
        json_path = Path(temp_dir) / "data.json"
        json_path.write_text('{"data": [1,2,3]}')

        with pytest.raises(ValueError, match="Unsupported file format"):
            loader.load_dataset(str(json_path))


# ============================================================================
# UTILITY TESTS
# ============================================================================


class TestUtilities:
    """Tests for utility functions."""

    def test_list_available_datasets(self):
        """Test listing available datasets."""
        available = DatasetLoader.list_available_datasets()

        assert "sklearn" in available
        assert "openml" in available
        assert "custom" in available

        assert len(available["sklearn"]) >= 5
        assert len(available["openml"]) >= 15

    def test_get_dataset_info_sklearn(self):
        """Test getting info for sklearn dataset."""
        info = DatasetLoader.get_dataset_info("iris")

        assert info["name"] == "iris"
        assert info["source"] == "sklearn"
        assert info["available"] is True

    def test_get_dataset_info_openml(self):
        """Test getting info for OpenML dataset."""
        info = DatasetLoader.get_dataset_info("credit_g")

        assert info["name"] == "credit_g"
        assert "openml" in info["source"].lower()
        assert info["available"] is True

    def test_get_dataset_info_unknown(self):
        """Test getting info for unknown dataset."""
        info = DatasetLoader.get_dataset_info("nonexistent_dataset")

        assert info["available"] is False

    def test_convenience_function(self):
        """Test load_benchmark_dataset convenience function."""
        data = load_benchmark_dataset("wine", test_size=0.2, random_state=42)

        assert "X_train" in data
        assert "y_train" in data
        assert data["metadata"]["n_samples"] == 178


# ============================================================================
# REPRODUCIBILITY TESTS
# ============================================================================


class TestReproducibility:
    """Tests for reproducibility with random seeds."""

    def test_same_seed_same_split(self, loader):
        """Test that same seed produces same split."""
        data1 = loader.load_dataset("iris", test_size=0.2, random_state=42)
        data2 = loader.load_dataset("iris", test_size=0.2, random_state=42)

        np.testing.assert_array_equal(data1["X_train"], data2["X_train"])
        np.testing.assert_array_equal(data1["y_train"], data2["y_train"])
        np.testing.assert_array_equal(data1["X_test"], data2["X_test"])
        np.testing.assert_array_equal(data1["y_test"], data2["y_test"])

    def test_different_seed_different_split(self, loader):
        """Test that different seeds produce different splits."""
        data1 = loader.load_dataset("iris", test_size=0.2, random_state=42)
        data2 = loader.load_dataset("iris", test_size=0.2, random_state=123)

        # Splits should be different
        assert not np.array_equal(data1["X_train"], data2["X_train"])


# ============================================================================
# EDGE CASES - NEW TESTS
# ============================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_single_feature_dataset(self, loader, temp_dir):
        """Test dataset with only 1 feature + target."""
        csv_path = Path(temp_dir) / "single_feature.csv"
        df = pd.DataFrame({"feature1": np.random.rand(20), "target": np.random.randint(0, 2, 20)})
        df.to_csv(csv_path, index=False)

        data = loader.load_dataset(str(csv_path), test_size=0.2, random_state=42)
        assert data["metadata"]["n_features"] == 1

    def test_large_feature_count(self, loader, temp_dir):
        """Test dataset with many features."""
        csv_path = Path(temp_dir) / "many_features.csv"
        n_features = 100
        n_samples = 50

        data_dict = {f"feat_{i}": np.random.rand(n_samples) for i in range(n_features)}
        data_dict["target"] = np.random.randint(0, 2, n_samples)
        df = pd.DataFrame(data_dict)
        df.to_csv(csv_path, index=False)

        data = loader.load_dataset(str(csv_path), test_size=0.2, random_state=42)
        assert data["metadata"]["n_features"] == n_features

    def test_all_same_values_feature(self, validator):
        """Test feature with all identical values."""
        X = np.array([[1.0, 5.0], [1.0, 5.0], [1.0, 5.0]])
        y = np.array([0, 1, 0])

        valid, warnings = validator.validate_dataset(X, y)
        assert valid
        assert any("zero variance" in w.lower() for w in warnings)

    def test_extreme_class_imbalance(self, validator):
        """Test extremely imbalanced dataset (99:1)."""
        X = np.random.rand(100, 4)
        y = np.concatenate([np.zeros(99), np.ones(1)])

        valid, warnings = validator.validate_dataset(X, y, task_type="classification")
        assert valid
        assert any("imbalance" in w.lower() for w in warnings)

    def test_multiclass_with_missing_class(self, loader, temp_dir):
        """Test multiclass where some classes might be missing after split."""
        csv_path = Path(temp_dir) / "rare_classes.csv"
        # Create dataset with very rare classes
        labels = [0] * 40 + [1] * 40 + [2] * 2  # Class 2 is very rare
        df = pd.DataFrame(
            {"feature1": np.random.rand(82), "feature2": np.random.rand(82), "target": labels}
        )
        df.to_csv(csv_path, index=False)

        # This might fail stratification or give warnings
        data = loader.load_dataset(str(csv_path), test_size=0.2, stratify=True, random_state=42)

        # Just check it doesn't crash
        assert data is not None


# ============================================================================
# ADDITIONAL VALIDATION TESTS
# ============================================================================


class TestAdditionalValidation:
    """Additional validation scenarios."""

    def test_clean_dataset_all_nan_column(self, validator):
        """Test cleaning dataset with entire column as NaN."""
        X = np.array([[1.0, np.nan], [2.0, np.nan], [3.0, np.nan]])
        y = np.array([0, 1, 0])

        X_clean, y_clean = validator.clean_dataset(X, y, strategy="median")

        # Should replace with median of non-NaN values (which is 0 for all-NaN)
        assert not np.any(np.isnan(X_clean))
        assert len(X_clean) == 3

    def test_clean_dataset_mixed_nan_inf(self, validator):
        """Test cleaning with both NaN and Inf."""
        X = np.array([[1.0, np.nan, np.inf], [2.0, 3.0, -np.inf], [4.0, 5.0, 6.0]])
        y = np.array([0, 1, 0])

        X_clean, y_clean = validator.clean_dataset(X, y, strategy="median")

        assert not np.any(np.isnan(X_clean))
        assert not np.any(np.isinf(X_clean))

    def test_validate_with_single_sample_class(self, validator):
        """Test validation warns about classes with single sample."""
        X = np.random.rand(20, 4)
        y = np.array([0] * 18 + [1, 2])  # Classes 1 and 2 have only 1 sample

        valid, warnings = validator.validate_dataset(X, y, task_type="classification")

        assert valid
        assert any("few samples" in w.lower() for w in warnings)


# ============================================================================
# PYTEST MARKERS
# ============================================================================

# To run only fast tests:
# pytest tests/unit/test_dataset_loader.py -v -m "not slow"

# To run only slow tests:
# pytest tests/unit/test_dataset_loader.py -v -m "slow"

# To run with coverage:
# pytest tests/unit/test_dataset_loader.py -v --cov=src/ga_trees/data --cov-report=html


"""
Pytest-compatible tests for Dataset Loader - FIXED VERSION

Usage:
    # Run all dataset loader tests
    pytest tests/unit/test_dataset_loader.py -v
    
    # Run with coverage
    pytest tests/unit/test_dataset_loader.py -v --cov=src.ga_trees.data
    
    # Run specific test
    pytest tests/unit/test_dataset_loader.py::TestSklearnDatasets::test_load_iris -v
    
    # Run only fast tests (skip slow OpenML tests)
    pytest tests/unit/test_dataset_loader.py -v -m "not slow"
"""
