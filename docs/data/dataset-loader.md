# Dataset Loader

`DatasetLoader` gets data from scikit-learn, OpenML or a local CSV/Excel file into one shape:
a train/test split plus a little metadata. Everything on this page matches
`src/ga_trees/data/dataset_loader.py`; if they ever disagree, the code wins.

## Quick start

```python
from ga_trees.data import DatasetLoader

loader = DatasetLoader()
data = loader.load_dataset("breast_cancer", test_size=0.2, standardize=True)

X_train, y_train = data["X_train"], data["y_train"]
X_test, y_test = data["X_test"], data["y_test"]

meta = data["metadata"]
print(
    f"{meta['n_samples']} samples, {meta['n_features']} features, {meta['n_classes']} classes"
)
```

`load_benchmark_dataset(name, **kwargs)` is a one-line shortcut for
`DatasetLoader().load_dataset(name, **kwargs)`.

## What it can load

**scikit-learn:** `iris`, `wine`, `breast_cancer`, `digits`, `diabetes` (regression).

**OpenML-CC18 benchmark set** — 20 datasets chosen for benchmarking, keyed by name in `DatasetLoader.CC18_BENCHMARK`:
`dresses_sales`, `kc2`, `climate_crashes`, `wdbc`, `ilpd`, `balance_scale`,
`credit_approval`, `breast_w`, `eucalyptus`, `blood_transfusion`, `diabetes_pima`,
`analcatdata_dmft`, `vehicle`, `tic_tac_toe`, `vowel`, `credit_g`, `qsar_biodeg`, `pc1`,
`banknote`, `pc4`.

**Other OpenML datasets** (not in the benchmark set): `heart`,
`mammography`, `ionosphere`, `sonar`, `hepatitis`, `titanic`, `adult`, `mnist`,
`credit_fraud`.

Any other name is tried against OpenML by name as a last resort. The full list, with IDs,
is in `DatasetLoader.OPENML_DATASETS`:

```python
from ga_trees.data import DatasetLoader

available = DatasetLoader.list_available_datasets()
print(available["sklearn"], available["openml"])
print(
    DatasetLoader.get_dataset_info("credit_g")
)  # {'name': ..., 'source': 'openml (ID: 31)', 'available': True}
```

**Files:** `.csv`, `.xlsx`, `.xls`. The last column is the target. String targets and
string feature columns are label-encoded to integers (not one-hot), so a categorical
feature becomes an ordinal one — fine for a tree's threshold splits, but worth knowing.

```python
data = loader.load_dataset("data/my_dataset.csv", test_size=0.2)
```

A name that looks like a path (has a `/`, or ends in a data-file extension) and doesn't
exist raises `FileNotFoundError` instead of being sent to OpenML.

## `load_dataset` options

```python
DatasetLoader(cache_dir=None)  # default cache: ~/.ga_trees/datasets

loader.load_dataset(
    name,
    test_size=0.2,  # fraction held out for testing
    random_state=42,  # seeds the split and any resampling
    stratify=True,  # stratified split; falls back to a random split if a class is too small
    standardize=False,  # StandardScaler fitted on the training split only
    balance=None,  # None, "oversample" or "undersample"
)
```

The return value is a dict:

| Key                 | Contents                                                                                                                       |
| ------------------- | ------------------------------------------------------------------------------------------------------------------------------ |
| `X_train`, `X_test` | Feature arrays                                                                                                                 |
| `y_train`, `y_test` | Label arrays                                                                                                                   |
| `feature_names`     | Column names                                                                                                                   |
| `target_names`      | Class names                                                                                                                    |
| `scaler`            | The fitted `StandardScaler`, or `None`                                                                                         |
| `metadata`          | `n_samples`, `n_features`, `n_classes`, `train_size`, `test_size`, `feature_names`, `target_names`, `balanced`, `standardized` |

There is no validation split in the return value. If you want the GA to score fitness on
held-out data (recommended — see [Training](../user-guides/training.md)), split
`X_train` yourself with `train_test_split`.

`balance` resamples the training split only, after the train/test split, so the test split
keeps the real class ratio and never contains copies of training rows. `train_size` in the
metadata is the size after resampling. If you'd rather not resample at all, a class-aware
metric does a similar job: `FitnessCalculator(classification_metric="balanced_accuracy")`.

## Validation and cleaning

Every load runs `DataValidator.validate_dataset` and logs (via `logging`, not `print`) any
warnings:

- NaN or Inf values in `X`
- fewer than 2 classes (raises)
- imbalance worse than 10:1
- a class with fewer than 5 samples
- zero-variance features

NaN/Inf values are then replaced with the column median. You can call the validator
directly:

```python
import numpy as np
from ga_trees.data import DataValidator

X = np.array([[1.0, np.nan], [2.0, 3.0], [3.0, 4.0]])
y = np.array([0, 1, 0])

ok, warnings = DataValidator.validate_dataset(X, y)
X_clean, y_clean = DataValidator.clean_dataset(
    X, y, strategy="median"
)  # or "mean", "remove", "zero"
```

Imputation runs on the whole dataset before the split, so test rows contribute to the
medians. That is a small leak; if it matters for your data, impute inside your own CV loop.

## Errors

| Situation                                      | Exception           |
| ---------------------------------------------- | ------------------- |
| Unknown name, not found on OpenML either       | `ValueError`        |
| Path-like name that doesn't exist              | `FileNotFoundError` |
| Empty file, one column, unsupported extension  | `ValueError`        |
| `balance` not `"oversample"` / `"undersample"` | `ValueError`        |
| Too few samples for the requested `test_size`  | `ValueError`        |
