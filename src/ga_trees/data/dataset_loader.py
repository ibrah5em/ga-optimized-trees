"""Load datasets from UCI and other sources."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import LabelEncoder


class DatasetLoader:
    """Load various datasets."""

    @staticmethod
    def load_uci_dataset(name: str):
        """Load UCI datasets via OpenML."""
        dataset_ids = {
            "credit": 31,  # Credit Approval
            "heart": 4,  # Heart Disease
            "diabetes": 37,  # Diabetes
            "ionosphere": 59,
            "sonar": 40,
            "hepatitis": 55,
        }

        if name not in dataset_ids:
            raise ValueError(f"Unknown dataset: {name}")

        print(f"Loading {name} from UCI/OpenML...")
        data = fetch_openml(data_id=dataset_ids[name], as_frame=True, parser="auto")

        X = data.data.values
        y = data.target.values

        # Encode labels if necessary
        if y.dtype == object:
            le = LabelEncoder()
            y = le.fit_transform(y)

        return X, y, data.feature_names, data.target_names

    @staticmethod
    def list_available_datasets():
        """List all available datasets."""
        return [
            "iris",
            "wine",
            "breast_cancer",  # Sklearn
            "credit",
            "heart",
            "diabetes",
            "ionosphere",
            "sonar",
            "hepatitis",  # UCI
        ]
