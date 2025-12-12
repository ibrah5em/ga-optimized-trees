

**Overview**
The enhanced dataset loader provides a unified interface for loading datasets from multiple sources with comprehensive validation, preprocessing, and augmentation capabilities specifically designed for genetic algorithm-based tree training.

**Features**
- **Multi-source Support**: Load from sklearn, OpenML (15+ benchmark datasets), CSV, and Excel files
- **Automatic Validation**: Detect missing values, class imbalance, zero-variance features
- **Preprocessing**: Optional standardization and balancing (oversample/undersample)
- **Automatic Splitting**: Stratified train/test splits for classification tasks
- **Robust Error Handling**: Clear error messages for malformed data or unsupported formats
- **Metadata Tracking**: Comprehensive dataset statistics and preprocessing state

**Quick Start**
```python
from ga_trees.data.dataset_loader import load_benchmark_dataset

# Load a built-in dataset with default settings
data = load_benchmark_dataset('titanic', 
                             test_size=0.2, 
                             standardize=True)

# Access train/test splits
X_train = data['X_train']
y_train = data['y_train']
X_test = data['X_test']
y_test = data['y_test']

# Access metadata
print(f"Dataset: {data['metadata']['name']}")
print(f"Features: {data['metadata']['n_features']}")
print(f"Classes: {data['metadata']['n_classes']}")
```

**Supported Datasets**

**Scikit-learn Datasets (5)**
- `iris`: Iris flowers (150 samples, 4 features, 3 classes)
- `wine`: Wine quality (178 samples, 13 features, 3 classes)
- `breast_cancer`: Breast cancer diagnosis (569 samples, 30 features, 2 classes)
- `digits`: Handwritten digits (1797 samples, 64 features, 10 classes)
- `diabetes`: Diabetes progression (442 samples, 10 features, regression task)

**OpenML Datasets (15+)**
- `credit_g`: German Credit (1000 samples, 20 features, 2 classes)
- `heart`: Heart Disease (303 samples, 13 features, 2 classes)
- `diabetes_pima`: Pima Indians Diabetes (768 samples, 8 features, 2 classes)
- `ionosphere`: Ionosphere (351 samples, 34 features, 2 classes)
- `sonar`: Sonar signals (208 samples, 60 features, 2 classes)
- `hepatitis`: Hepatitis (155 samples, 19 features, 2 classes)
- `titanic`: Titanic survival (2201 samples, 11 features, 2 classes)
- `adult`: Adult income >50K (48842 samples, 14 features, 2 classes)
- `mnist`: MNIST digits (70000 samples, 784 features, 10 classes)
- `credit_fraud`: Credit card fraud (284807 samples, 30 features, 2 classes)
- `vehicle`: Vehicle silhouettes (846 samples, 18 features, 4 classes)
- `balance_scale`: Balance scale (625 samples, 4 features, 3 classes)
- `blood_transfusion`: Blood transfusion (748 samples, 4 features, 2 classes)
- `banknote`: Banknote authentication (1372 samples, 4 features, 2 classes)
- `mammographic`: Mammographic mass (961 samples, 5 features, 2 classes)

**Custom Files**
- CSV files (comma-separated values)
- Excel files (.xlsx, .xls formats)

**Usage Examples**

**1. Basic Dataset Loading**
```python
from ga_trees.data.dataset_loader import DatasetLoader

loader = DatasetLoader()

# Load Titanic dataset with default 80/20 split
data = loader.load_dataset('titanic', test_size=0.2)

print(f"Train size: {data['metadata']['train_size']}")
print(f"Test size: {data['metadata']['test_size']}")
print(f"Features: {data['feature_names'][:5]}...")  # First 5 features
```

**2. Load with Feature Standardization**
```python
# Standardize features to zero mean and unit variance
data = loader.load_dataset('diabetes_pima', 
                          test_size=0.3,
                          standardize=True)

# Scaler object is saved for transforming new data
scaler = data['scaler']
print(f"Scaler type: {type(scaler)}")
print(f"Mean values: {scaler.mean_}")
```

**3. Handle Imbalanced Data**
```python
# Oversample minority class using SMOTE
data = loader.load_dataset('credit_fraud',
                          test_size=0.2,
                          balance='oversample')

# Undersample majority class using RandomUnderSampler
data = loader.load_dataset('credit_fraud',
                          test_size=0.2,
                          balance='undersample')

# Check class distribution after balancing
from collections import Counter
print(f"Train class distribution: {Counter(data['y_train'])}")
```

**4. Load Custom CSV File**
```python
# CSV format requirements:
# - Last column must be target variable
# - First row can be column names (optional)
# Example format:
# feature1,feature2,feature3,target
# 1.2,3.4,5.6,0
# 2.3,4.5,6.7,1

data = loader.load_dataset('data/my_dataset.csv',
                          test_size=0.2,
                          standardize=True)
```

**5. Load Excel File with Specific Sheet**
```python
# Load specific sheet from Excel file
data = loader.load_dataset('data/experiment_results.xlsx',
                          sheet_name='Sheet1',  # Optional parameter
                          test_size=0.25,
                          stratify=True)
```

**6. Advanced Configuration with All Options**
```python
data = loader.load_dataset(
    name='adult',
    test_size=0.3,           # 30% for testing, 70% for training
    random_state=42,         # Reproducible random splits
    stratify=True,           # Maintain class proportions in split
    standardize=True,        # Scale features using StandardScaler
    balance='oversample',    # Handle class imbalance
    validation_split=0.1,    # Optional: further split train into train/val
    verbose=True             # Print loading progress and warnings
)
```

**7. List All Available Datasets**
```python
# Get dictionary of all available datasets by source
available = DatasetLoader.list_available_datasets()

print("Sklearn datasets:", available['sklearn'])
print("\nOpenML datasets:", available['openml'][:5], "...")  # First 5
print(f"\nTotal datasets: {len(available['sklearn']) + len(available['openml'])}")

# Check if specific dataset is available
if 'titanic' in available['openml']:
    print("Titanic dataset is available")
```

**8. Get Detailed Dataset Information**
```python
info = DatasetLoader.get_dataset_info('titanic')
print(f"Dataset: {info['name']}")
print(f"Source: {info['source']}")
print(f"Available: {info['available']}")
print(f"Description: {info.get('description', 'No description available')}")
```

**Integration with GA Training**

**Example: Train GA-Tree Model with Loaded Dataset**
```python
from ga_trees.data.dataset_loader import load_benchmark_dataset
from ga_trees.ga.engine import GAEngine, GAConfig, TreeInitializer, Mutation
from ga_trees.fitness.calculator import FitnessCalculator

# Load and preprocess dataset
data = load_benchmark_dataset('heart', 
                             test_size=0.2, 
                             standardize=True)

X_train = data['X_train']
y_train = data['y_train']
X_test = data['X_test']
y_test = data['y_test']

# Extract metadata for GA configuration
n_features = data['metadata']['n_features']
n_classes = data['metadata']['n_classes']
feature_ranges = {i: (X_train[:, i].min(), X_train[:, i].max()) 
                 for i in range(n_features)}

# Configure GA parameters
ga_config = GAConfig(
    population_size=80,
    n_generations=40,
    crossover_rate=0.8,
    mutation_rate=0.2
)

# Initialize components
initializer = TreeInitializer(
    n_features=n_features,
    n_classes=n_classes,
    max_depth=6,
    min_samples_split=8,
    min_samples_leaf=3
)

fitness_calc = FitnessCalculator()
mutation = Mutation(
    n_features=n_features,
    feature_ranges=feature_ranges
)

# Create and run GA engine
ga_engine = GAEngine(
    config=ga_config,
    initializer=initializer,
    fitness_func=fitness_calc.calculate_fitness,
    mutation=mutation
)

best_tree = ga_engine.evolve(
    X_train, 
    y_train, 
    verbose=True,
    early_stopping_patience=10
)

# Evaluate on test set
from ga_trees.fitness.calculator import TreePredictor
from sklearn.metrics import accuracy_score, classification_report

predictor = TreePredictor()
y_pred = predictor.predict(best_tree, X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"\nTest Accuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
```

**Data Validation**
The loader automatically performs quality checks and provides warnings:

```python
data = loader.load_dataset('ionosphere')
# Output example:
# ⚠ Warning: Found 16 features with zero variance
# ⚠ Warning: Class ratio is 64:36 (minority class < 40%)
# ✓ Loaded: 351 samples, 34 features, 2 classes
```

**Validation checks include:**
1. Missing values detection (NaN, Inf)
2. Empty or too small datasets (<10 samples)
3. Minimum class size validation (warns if any class < 5 samples)
4. Severe class imbalance detection (>10:1 ratio)
5. Zero variance feature detection
6. Feature-target dimension consistency
7. Data type validation (converts non-numeric where possible)

**Error Handling**
```python
try:
    # Non-existent dataset
    data = loader.load_dataset('nonexistent_dataset')
except ValueError as e:
    print(f"Error: {e}")
    # Error: Unknown dataset: 'nonexistent_dataset'. Available datasets: [...]

try:
    # Malformed file
    data = loader.load_dataset('malformed.csv')
except FileNotFoundError as e:
    print(f"Error: {e}")
    # Error: File not found: 'malformed.csv'
except ValueError as e:
    print(f"Error: {e}")
    # Error: CSV file must have at least 2 columns (features + target)

try:
    # Invalid parameter
    data = loader.load_dataset('iris', test_size=1.5)
except ValueError as e:
    print(f"Error: {e}")
    # Error: test_size must be between 0.0 and 1.0
```

**CSV File Format Requirements**
For custom CSV files, follow this exact format:

```csv
feature_1,feature_2,feature_3,target
1.2,3.4,5.6,0
2.3,4.5,6.7,1
3.4,5.6,7.8,0
```

**Rules:**
1. First row: column names (optional but recommended)
2. Last column: target variable (must be numeric or categorical)
3. All other columns: features (numeric or categorical)
4. Missing values: represented as empty strings or NA
5. Categorical features: automatically one-hot encoded
6. Categorical targets: automatically label encoded to integers
7. File encoding: UTF-8 recommended

**Return Value Structure**
```python
data = loader.load_dataset('iris')

# Returns dictionary with following structure:
{
    # Core data splits
    'X_train': np.ndarray,      # Training features (n_train_samples, n_features)
    'X_test': np.ndarray,       # Test features (n_test_samples, n_features)
    'y_train': np.ndarray,      # Training labels (n_train_samples,)
    'y_test': np.ndarray,       # Test labels (n_test_samples,)
    
    # Optional validation split (if validation_split > 0)
    'X_val': np.ndarray,        # Validation features
    'y_val': np.ndarray,        # Validation labels
    
    # Feature and target information
    'feature_names': List[str], # Original feature names
    'target_names': List[str],  # Original class names (if categorical)
    
    # Preprocessing objects
    'scaler': StandardScaler,   # Fitted scaler (if standardize=True)
    'balancer': object,         # Balancing object (if balance != None)
    
    # Comprehensive metadata
    'metadata': {
        'name': str,            # Dataset name
        'source': str,          # Data source ('sklearn', 'openml', 'file')
        'n_samples': int,       # Total samples before split
        'n_features': int,      # Number of features
        'n_classes': int,       # Number of classes (1 for regression)
        'train_size': int,      # Training samples count
        'test_size': int,       # Test samples count
        'val_size': int,        # Validation samples count (if any)
        'balanced': bool,       # Whether balancing was applied
        'standardized': bool,   # Whether standardization was applied
        'task_type': str,       # 'classification' or 'regression'
        'class_distribution': dict,  # Original class counts
        'feature_types': dict,  # Feature data types
        'warnings': List[str]   # Any validation warnings
    }
}
```

**Best Practices**

**1. Always Check Data Quality**
```python
data = loader.load_dataset('my_data.csv')
warnings = data['metadata']['warnings']
if warnings:
    print("Data quality warnings:")
    for warning in warnings:
        print(f"  ⚠ {warning}")
```

**2. Use Stratification for Classification**
```python
# Maintains class proportions in train/test splits
data = loader.load_dataset('heart', stratify=True)

# Verify stratification worked
from collections import Counter
train_dist = Counter(data['y_train'])
test_dist = Counter(data['y_test'])
print(f"Train distribution: {train_dist}")
print(f"Test distribution: {test_dist}")
```

**3. Standardize for Better GA Performance**
```python
# Genetic algorithms converge faster with standardized features
data = loader.load_dataset('diabetes', standardize=True)

# Save scaler for consistent transformation of new data
import joblib
joblib.dump(data['scaler'], 'diabetes_scaler.pkl')
```

**4. Handle Imbalanced Data Appropriately**
```python
# For severe imbalance (> 3:1 ratio), use balancing
data = loader.load_dataset('credit_fraud', balance='oversample')

# Compare strategies
strategies = [None, 'oversample', 'undersample']
for strategy in strategies:
    data = loader.load_dataset('credit_fraud', balance=strategy)
    dist = Counter(data['y_train'])
    print(f"{strategy}: {dist}")
```

**5. Ensure Reproducibility**
```python
# Set random seeds for reproducible splits
data1 = loader.load_dataset('wine', random_state=42)
data2 = loader.load_dataset('wine', random_state=42)

# Verify reproducibility
assert np.array_equal(data1['X_train'], data2['X_train'])
assert np.array_equal(data1['y_train'], data2['y_train'])
```

**Performance Tips**

**For Large Datasets (>10K samples):**
```python
# Use smaller test size to reduce training time
data = loader.load_dataset('mnist', test_size=0.1)

# Consider sampling for initial experimentation
from sklearn.utils import resample
X_sampled, y_sampled = resample(data['X_train'], data['y_train'], 
                                n_samples=5000, random_state=42)
```

**For High-Dimensional Data (>100 features):**
```python
# Load without preprocessing first
data = loader.load_dataset('high_dim_data.csv', standardize=False)

# Apply feature selection before training
from sklearn.feature_selection import VarianceThreshold
selector = VarianceThreshold(threshold=0.01)
X_train_selected = selector.fit_transform(data['X_train'])
```

**For Very Imbalanced Data:**
```python
# Try different balancing strategies
strategies = {
    'none': None,
    'oversample': 'oversample',
    'undersample': 'undersample',
    'smote': 'oversample'  # Uses SMOTE by default
}

results = {}
for name, strategy in strategies.items():
    data = loader.load_dataset('fraud', balance=strategy)
    # Train and evaluate...
```

**Troubleshooting**

**Issue: "OpenML dataset not found"**
```python
# Solution 1: Check exact dataset name
available = DatasetLoader.list_available_datasets()
print("Available OpenML datasets:", available['openml'])

# Solution 2: Load by OpenML ID directly
from ga_trees.data.dataset_loader import _load_openml_by_id
data = _load_openml_by_id(31)  # German Credit dataset ID
```

**Issue: "NaN values detected in dataset"**
```python
# Solution 1: Automatic cleaning during load
data = loader.load_dataset('data_with_nans.csv')

# Solution 2: Manual cleaning before loading
from ga_trees.data.validator import DataValidator
validator = DataValidator()
X_clean, y_clean = validator.clean_dataset(X, y, strategy='mean')
```

**Issue: "Severe class imbalance detected"**
```python
# Solution: Apply balancing during load
data = loader.load_dataset('imbalanced_data.csv', balance='oversample')

# Or adjust class weights in GA training
ga_config = GAConfig(
    population_size=100,
    class_weight='balanced'  # Use class weights in fitness calculation
)
```

**Issue: "CSV loading fails with encoding error"**
```python
# Solution: Specify file encoding
data = loader.load_dataset('data.csv', encoding='latin-1')

# Or clean the CSV file first
import pandas as pd
df = pd.read_csv('data.csv', encoding='utf-8', errors='ignore')
df.to_csv('data_clean.csv', index=False)
data = loader.load_dataset('data_clean.csv')
```

**Testing Your Dataset**
```python
from ga_trees.data.dataset_loader import DatasetLoader
import numpy as np

loader = DatasetLoader()

try:
    # Load and validate dataset
    data = loader.load_dataset('my_custom_data.csv', verbose=True)
    
    print("✓ Dataset loaded successfully!")
    print(f"  Samples: {data['metadata']['n_samples']}")
    print(f"  Features: {data['metadata']['n_features']}")
    print(f"  Classes: {data['metadata']['n_classes']}")
    
    # Check class distribution
    unique, counts = np.unique(data['y_train'], return_counts=True)
    print(f"  Train class distribution: {dict(zip(unique, counts))}")
    
    unique, counts = np.unique(data['y_test'], return_counts=True)
    print(f"  Test class distribution: {dict(zip(unique, counts))}")
    
    # Check for any issues
    if data['metadata']['warnings']:
        print("\n⚠ Warnings:")
        for warning in data['metadata']['warnings']:
            print(f"  - {warning}")
            
except Exception as e:
    print(f"✗ Error loading dataset: {e}")
    print("\nDebugging steps:")
    print("1. Check file exists and is accessible")
    print("2. Verify CSV format (last column is target)")
    print("3. Check for missing values or invalid data")
    print("4. Ensure all features are numeric or can be converted")
```

**API Reference**

**DatasetLoader Class**

*Constructor:*
```python
DatasetLoader(cache_dir=None, download_dir=None)
```
- `cache_dir`: Directory to cache downloaded datasets (default: ~/.ga_trees/cache)
- `download_dir`: Directory for downloaded files (default: ~/.ga_trees/downloads)

*Methods:*

**load_dataset(name, test_size=0.2, random_state=42, stratify=True, standardize=False, balance=None, validation_split=0.0, verbose=False, **kwargs)**
- `name`: Dataset name, OpenML ID, or file path
- `test_size`: Proportion of data for testing (0.0 to 1.0)
- `random_state`: Random seed for reproducibility
- `stratify`: Maintain class distribution in splits (classification only)
- `standardize`: Scale features to zero mean and unit variance
- `balance`: 'oversample', 'undersample', or None
- `validation_split`: Additional validation split from training data
- `verbose`: Print loading progress and warnings
- Returns: Dictionary with data splits and metadata

**list_available_datasets()**
- Returns: Dictionary with keys 'sklearn' and 'openml' containing lists of available datasets

**get_dataset_info(name)**
- `name`: Dataset name to query
- Returns: Dictionary with dataset metadata including availability

**DataValidator Class**

*Methods:*

**validate_dataset(X, y, task_type='classification')**
- `X`: Feature matrix
- `y`: Target vector
- `task_type`: 'classification' or 'regression'
- Returns: Tuple of (is_valid, warnings_list)

**clean_dataset(X, y, strategy='remove')**
- `X`: Feature matrix (may contain NaN/Inf)
- `y`: Target vector
- `strategy`: 'remove' (drop rows), 'mean' (fill with mean), 'median' (fill with median)
- Returns: Tuple of (X_cleaned, y_cleaned)

**check_class_balance(y, threshold=0.1)**
- `y`: Target vector
- `threshold`: Minimum minority class proportion
- Returns: Tuple of (is_balanced, balance_ratio, warnings)

**Examples Repository**
See `scripts/dataset_examples.py` for comprehensive examples including:
- Loading and comparing all benchmark datasets
- Custom preprocessing pipelines
- Integration with experiment tracking
- Batch processing multiple datasets
- Cross-validation setups with GA-Trees

**Version Compatibility**
- Python 3.8+
- scikit-learn >= 1.0.0
- pandas >= 1.3.0
- openml >= 0.12.0
- imbalanced-learn >= 0.9.0 (for balancing)
- numpy >= 1.21.0

**License and Attribution**
- Built-in datasets follow their original licenses
- OpenML datasets: CC-BY 4.0 unless otherwise specified
- Always cite original dataset sources in publications