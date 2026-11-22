from dataclasses import dataclass
import numpy as np
import pandas as pd

@dataclass
class ClusteringPreprocessor:
    preprocessor: object
    pca: object
    kmeans: object

    def transform(self, X: pd.DataFrame):
        X_scaled = self.preprocessor.transform(X)
        X_reduced = self.pca.transform(X_scaled)
        return X_reduced

    def fit_transform(self, X: pd.DataFrame):
        X_scaled = self.preprocessor.fit_transform(X)
        X_reduced = self.pca.fit_transform(X_scaled)
        self.kmeans.fit(X_reduced)
        return X_reduced

    def predict_cluster(self, X: pd.DataFrame):
        return self.kmeans.predict(self.transform(X))

class CustomerSegmentationModel:
    def __init__(self, clustering_preprocessor: ClusteringPreprocessor, classifier: object):
        self.clustering_preprocessor = clustering_preprocessor
        self.classifier = classifier

    def predict(self, X: pd.DataFrame):
        X_reduced = self.clustering_preprocessor.transform(X)
        return self.classifier.predict(X_reduced)

    def predict_proba(self, X: pd.DataFrame):
        X_reduced = self.clustering_preprocessor.transform(X)
        return self.classifier.predict_proba(X_reduced)
