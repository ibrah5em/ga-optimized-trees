Dataset Loader Documentation
Overview
The enhanced dataset loader supports multiple data sources with comprehensive validation, preprocessing, and augmentation capabilities.

Features
✓ 15+ benchmark datasets from sklearn, OpenML, and UCI
✓ CSV/Excel file loading with automatic type detection
✓ Data validation - detect missing values, imbalance, zero-variance features
✓ Automatic splitting - stratified train/test splits
✓ Balancing - oversample/undersample for imbalanced data
✓ Standardization - optional feature scaling
✓ Error handling - robust error messages for malformed data

Quick Start
python
from ga_trees.data.dataset_loader import load_benchmark_dataset

# Load a built-in dataset
data = load_benchmark_dataset('titanic', 
                             test_size=0.2, 
                             standardize=True)

X_train = data['X_train']
y_train = data['y_train']
X_test = data['X_test']
y_test = data['y_test']
Supported Datasets
Scikit-learn Datasets (5)
iris - Iris flowers (150 samples, 4 features, 3 classes)
wine - Wine quality (178 samples, 13 features, 3 classes)
breast_cancer - Breast cancer diagnosis (569 samples, 30 features, 2 classes)
digits - Handwritten digits (1797 samples, 64 features, 10 classes)
diabetes - Diabetes progression (442 samples, 10 features, regression)
OpenML Datasets (15+)
credit_g - German Credit (1000 samples, 20 features, 2 classes)
heart - Heart Disease (303 samples, 13 features, 2 classes)
diabetes_pima - Pima Indians Diabetes (768 samples, 8 features, 2 classes)
ionosphere - Ionosphere (351 samples, 34 features, 2 classes)
sonar - Sonar signals (208 samples, 60 features, 2 classes)
hepatitis - Hepatitis (155 samples, 19 features, 2 classes)
titanic - Titanic survival (2201 samples, 11 features, 2 classes)
adult - Adult income >50K (48842 samples, 14 features, 2 classes)
mnist - MNIST digits (70000 samples, 784 features, 10 classes)
credit_fraud - Credit card fraud (284807 samples, 30 features, 2 classes)
vehicle - Vehicle silhouettes (846 samples, 18 features, 4 classes)
balance_scale - Balance scale (625 samples, 4 features, 3 classes)
blood_transfusion - Blood transfusion (748 samples, 4 features, 2 classes)
banknote - Banknote authentication (1372 samples, 4 features, 2 classes)
mammographic - Mammographic mass (961 samples, 5 features, 2 classes)
Custom Files
CSV files - Comma-separated values
Excel files - .xlsx, .xls formats
Usage Examples
1. Basic Loading
python
from ga_trees.data.dataset_loader import DatasetLoader

loader = DatasetLoader()

# Load Titanic dataset
data = loader.load_dataset('titanic', test_size=0.2)

print(f"Train size: {data['metadata']['train_size']}")
print(f"Test size: {data['metadata']['test_size']}")
print(f"Features: {data['feature_names']}")
2. Load with Standardization
python
# Standardize features for better convergence
data = loader.load_dataset('diabetes_pima', 
                          test_size=0.3,
                          standardize=True)

# Scaler is saved for later use
scaler = data['scaler']
X_new_standardized = scaler.transform(X_new)
3. Handle Imbalanced Data
python
# Oversample minority class
data = loader.load_dataset('credit_fraud',
                          test_size=0.2,
                          balance='oversample')

# Or undersample majority class
data = loader.load_dataset('credit_fraud',
                          test_size=0.2,
                          balance='undersample')
4. Load Custom CSV
python
# CSV format: last column is target
# Example: feature1,feature2,feature3,target
#          1.2,3.4,5.6,0
#          2.3,4.5,6.7,1

data = loader.load_dataset('data/my_dataset.csv',
                          test_size=0.2,
                          standardize=True)
5. Load Excel File
python
data = loader.load_dataset('data/experiment_results.xlsx',
                          test_size=0.25,
                          stratify=True)
6. Advanced Configuration
python
data = loader.load_dataset(
    name='adult',
    test_size=0.3,           # 30% for testing
    random_state=42,         # Reproducible splits
    stratify=True,           # Maintain class proportions
    standardize=True,        # Scale features
    balance='oversample'     # Handle imbalance
)
7. List Available Datasets
python
available = DatasetLoader.list_available_datasets()

print("Sklearn datasets:", available['sklearn'])
print("OpenML datasets:", available['openml'])
8. Get Dataset Info
python
info = DatasetLoader.get_dataset_info('titanic')
print(info)
# {'name': 'titanic', 'source': 'openml (ID: 40945)', 'available': True}
Integration with GA Training
Example: Train with Custom Dataset
python
from ga_trees.data.dataset_loader import load_benchmark_dataset
from ga_trees.ga.engine import GAEngine, GAConfig, TreeInitializer, Mutation
from ga_trees.fitness.calculator import FitnessCalculator

# Load dataset
data = load_benchmark_dataset('heart', 
                             test_size=0.2, 
                             standardize=True)

X_train = data['X_train']
y_train = data['y_train']
X_test = data['X_test']
y_test = data['y_test']

# Setup GA
n_features = data['metadata']['n_features']
n_classes = data['metadata']['n_classes']
feature_ranges = {i: (X_train[:, i].min(), X_train[:, i].max()) 
                 for i in range(n_features)}

ga_config = GAConfig(population_size=80, n_generations=40)
initializer = TreeInitializer(n_features=n_features, n_classes=n_classes,
                             max_depth=6, min_samples_split=8, min_samples_leaf=3)
fitness_calc = FitnessCalculator()
mutation = Mutation(n_features=n_features, feature_ranges=feature_ranges)

# Train
ga_engine = GAEngine(ga_config, initializer, 
                    fitness_calc.calculate_fitness, mutation)
best_tree = ga_engine.evolve(X_train, y_train, verbose=True)

# Evaluate
from ga_trees.fitness.calculator import TreePredictor
predictor = TreePredictor()
y_pred = predictor.predict(best_tree, X_test)

from sklearn.metrics import accuracy_score, classification_report
accuracy = accuracy_score(y_test, y_pred)
print(f"\nTest Accuracy: {accuracy:.4f}")
print(classification_report(y_test, y_pred))
Data Validation
The loader automatically validates data quality:

python
data = loader.load_dataset('ionosphere')
# ⚠ Warnings: ['Found 16 features with zero variance']
# ✓ Loaded: 351 samples, 34 features, 2 classes
Validation checks:

Missing values (NaN, Inf)
Empty datasets
Minimum class sizes
Severe class imbalance (>10:1)
Zero variance features
Dimension mismatches
Error Handling
python
try:
    data = loader.load_dataset('nonexistent_dataset')
except ValueError as e:
    print(f"Error: {e}")
    # Error: Unknown dataset: nonexistent_dataset

try:
    data = loader.load_dataset('malformed.csv')
except FileNotFoundError as e:
    print(f"Error: {e}")
    # Error: File not found: malformed.csv
CSV File Format Requirements
For custom CSV files:

csv
feature_1,feature_2,feature_3,target
1.2,3.4,5.6,0
2.3,4.5,6.7,1
3.4,5.6,7.8,0
Rules:

First row: column names (optional)
Last column: target variable
Numeric or categorical features (auto-encoded)
Categorical targets are automatically encoded to integers
Return Value Structure
python
data = loader.load_dataset('iris')

# Dictionary with keys:
{
    'X_train': np.ndarray,      # Training features
    'X_test': np.ndarray,       # Test features
    'y_train': np.ndarray,      # Training labels
    'y_test': np.ndarray,       # Test labels
    'feature_names': List[str], # Feature names
    'target_names': List[str],  # Class names
    'scaler': StandardScaler,   # Scaler (if standardize=True)
    'metadata': {
        'n_samples': int,
        'n_features': int,
        'n_classes': int,
        'train_size': int,
        'test_size': int,
        'balanced': bool,
        'standardized': bool
    }
}
Best Practices
1. Always Validate Data
python
# Validation is automatic, but check warnings
data = loader.load_dataset('my_data.csv')
# Review any warnings printed
2. Use Stratification
python
# For classification, maintain class proportions
data = loader.load_dataset('heart', stratify=True)
3. Standardize for GA
python
# GA works better with standardized features
data = loader.load_dataset('diabetes', standardize=True)
4. Handle Imbalanced Data
python
# If class imbalance > 3:1, consider balancing
data = loader.load_dataset('credit_fraud', balance='oversample')
5. Set Random Seed
python
# For reproducibility
data = loader.load_dataset('wine', random_state=42)
Performance Tips
Large Datasets (>10K samples):

python
# Use smaller test_size to speed up training
data = loader.load_dataset('mnist', test_size=0.1)
High-Dimensional Data (>100 features):

python
# Consider feature selection first
from sklearn.feature_selection import SelectKBest
# ... then load and train
Imbalanced Data:

python
# Try both strategies and compare
data_over = loader.load_dataset('fraud', balance='oversample')
data_under = loader.load_dataset('fraud', balance='undersample')
Troubleshooting
Issue: "OpenML dataset not found"
Solution: Check dataset name spelling or try by ID:

python
data = loader._load_openml_by_id(31)  # German Credit
Issue: "NaN values detected"
Solution: Clean data automatically:

python
# Use 'mean' strategy to fill NaN with column means
X, y = loader.validator.clean_dataset(X, y, strategy='mean')
Issue: "Severe class imbalance"
Solution: Balance the dataset:

python
data = loader.load_dataset('fraud', balance='oversample')
Issue: CSV loading fails
Solution: Check file format:

Ensure last column is target
Remove any extra headers/footers
Check for missing values
Testing Your Dataset
python
# Test if dataset loads correctly
from ga_trees.data.dataset_loader import DatasetLoader

loader = DatasetLoader()

try:
    data = loader.load_dataset('my_custom_data.csv')
    
    print("✓ Dataset loaded successfully!")
    print(f"  Samples: {data['metadata']['n_samples']}")
    print(f"  Features: {data['metadata']['n_features']}")
    print(f"  Classes: {data['metadata']['n_classes']}")
    
    # Check class distribution
    import numpy as np
    unique, counts = np.unique(data['y_train'], return_counts=True)
    print(f"  Class distribution: {dict(zip(unique, counts))}")
    
except Exception as e:
    print(f"✗ Error: {e}")
API Reference
DatasetLoader
Methods:

load_dataset(name, test_size=0.2, random_state=42, stratify=True, standardize=False, balance=None)
Load dataset from any source
Returns: Dictionary with train/test data
list_available_datasets()
List all available datasets
Returns: Dictionary of dataset sources
get_dataset_info(name)
Get metadata about a dataset
Returns: Dictionary with dataset info
DataValidator
Methods:

validate_dataset(X, y, task_type='classification')
Validate data quality
Returns: (is_valid, warnings)
clean_dataset(X, y, strategy='remove')
Clean missing/invalid values
Returns: (X_cleaned, y_cleaned)
Examples Repository
See scripts/dataset_examples.py for more examples:

Loading all benchmark datasets
Comparing dataset characteristics
Custom preprocessing pipelines
Integration with experiment workflows
