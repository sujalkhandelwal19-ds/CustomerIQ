import sys
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from src.constant.training_pipeline import TARGET_COLUMN
from src.exception import CustomerException
from src.logger import logging


class CreateClusters:
    """Fit PCA + KMeans once on training data and reuse them for test/prediction."""

    def __init__(self, n_clusters: int = 3, n_components: int = 2, random_state: int = 42):
        self.n_clusters = n_clusters
        self.n_components = n_components
        self.random_state = random_state
        self.pca = PCA(n_components=n_components, random_state=random_state)
        self.model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
        self.is_fitted = False

    def fit(self, preprocessed_data: pd.DataFrame) -> pd.DataFrame:
        try:
            reduced = self.pca.fit_transform(preprocessed_data)
            labels = self.model.fit_predict(reduced)
            self.is_fitted = True
            result = preprocessed_data.copy()
            result[TARGET_COLUMN] = labels.astype(int)
            logging.info("PCA and KMeans fitted on training data.")
            return result
        except Exception as e:
            raise CustomerException(e, sys) from e

    def transform(self, preprocessed_data: pd.DataFrame) -> pd.DataFrame:
        try:
            if not self.is_fitted:
                raise ValueError("Clusterer must be fitted before transform().")
            reduced = self.pca.transform(preprocessed_data)
            result = preprocessed_data.copy()
            result[TARGET_COLUMN] = self.model.predict(reduced).astype(int)
            return result
        except Exception as e:
            raise CustomerException(e, sys) from e

    def fit_transform(self, preprocessed_data: pd.DataFrame):
        return self.fit(preprocessed_data)

    def get_dataset_using_pca(self, preprocessed_data: pd.DataFrame):
        """Backward-compatible helper: fit PCA only and return reduced data."""
        try:
            return self.pca.fit_transform(preprocessed_data)
        except Exception as e:
            raise CustomerException(e, sys) from e

    def initialize_clustering(self, preprocessed_data: pd.DataFrame) -> pd.DataFrame:
        """Backward-compatible alias for fitting clustering."""
        return self.fit(preprocessed_data)
