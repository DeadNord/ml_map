from itertools import product
from sklearn.base import clone
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
)
import numpy as np
import pandas as pd


class ULModelTrainer:
    """
    A class to create and train model pipelines with grid search for hyperparameter tuning.

    Attributes
    ----------
    best_model : object
        The best model found by grid search.
    best_params : dict
        The best hyperparameters found by grid search.
    best_score : float
        The best score achieved by grid search.
    best_model_name : str
        Name of the best model based on the evaluation score.

    Methods
    -------
    train(X_train, pipelines, param_grids, scoring='silhouette_score'):
        Trains the pipelines with grid search.
    predict(X_test):
        Predicts cluster labels for the given test data using the best model.
    """

    def __init__(self):
        self.best_model = None
        self.best_params = None
        self.best_score = float("-inf")
        self.best_model_name = None

    def train(self, X_train, pipelines, param_grids, scoring="silhouette_score"):
        for model_name, pipeline in pipelines.items():
            param_grid = param_grids[model_name]

            param_combinations = list(product(*param_grid.values()))

            for params in param_combinations:
                param_dict = {
                    param_name: param_value
                    for param_name, param_value in zip(param_grid.keys(), params)
                }
                model = clone(pipeline)
                model.set_params(**param_dict)
                model.fit(X_train)
                cluster_labels = model.predict(X_train)

                if scoring == "silhouette_score":
                    score = silhouette_score(X_train, cluster_labels)
                elif scoring == "davies_bouldin_score":
                    score = -davies_bouldin_score(X_train, cluster_labels)
                elif scoring == "calinski_harabasz_score":
                    score = calinski_harabasz_score(X_train, cluster_labels)
                else:
                    raise ValueError(f"Unsupported scoring method: {scoring}")

                if score > self.best_score:
                    self.best_score = score
                    self.best_model = model
                    self.best_params = param_dict
                    self.best_model_name = model_name

    def predict(self, X_test):
        return self.best_model.predict(X_test)
