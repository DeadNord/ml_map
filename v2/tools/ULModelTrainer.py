from sklearn.base import clone, BaseEstimator, ClusterMixin
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
)
import numpy as np


class ClusterEstimatorWrapper(BaseEstimator, ClusterMixin):
    """
    A wrapper to adapt clustering models for use with GridSearchCV.

    This wrapper allows clustering models to be used with GridSearchCV by defining a dummy fit method.
    """

    def __init__(self, estimator, **kwargs):
        self.estimator = estimator
        self.kwargs = kwargs

    def fit(self, X, y=None):
        self.estimator.fit(X)
        return self

    def predict(self, X):
        return self.estimator.fit_predict(X)

    def fit_predict(self, X, y=None):
        return self.estimator.fit_predict(X)

    def get_params(self, deep=True):
        return {"estimator": self.estimator, **self.kwargs}

    def set_params(self, **params):
        if "estimator" in params:
            self.estimator = params.pop("estimator")
        self.kwargs.update(params)
        self.estimator.set_params(**self.kwargs)
        return self


# Custom scorer functions that only use X for scoring
def silhouette_scorer(estimator, X, y=None):
    labels = estimator.fit_predict(X)
    return silhouette_score(X, labels)


def davies_bouldin_scorer(estimator, X, y=None):
    labels = estimator.fit_predict(X)
    return -davies_bouldin_score(
        X, labels
    )  # Note: Lower is better, hence the negative sign


def calinski_harabasz_scorer(estimator, X, y=None):
    labels = estimator.fit_predict(X)
    return calinski_harabasz_score(X, labels)


class ULModelTrainer:
    """
    A class to create and train model pipelines with grid search for hyperparameter tuning.

    Attributes
    ----------
    best_models : dict
        Dictionary of best models found by grid search for each model.
    best_params : dict
        Dictionary of best hyperparameters found by grid search for each model.
    best_scores : dict
        Dictionary of best scores achieved by grid search for each model.
    best_model_name : str
        Name of the best model based on the evaluation score.

    Methods
    -------
    train(X_train, X_val, pipelines, param_grids, scoring='silhouette_score'):
        Trains the pipelines with grid search.
    predict(X_test):
        Predicts cluster labels for the given test data using the best model.
    """

    def __init__(self):
        self.best_models = {}
        self.best_params = {}
        self.best_scores = {}
        self.best_model_name = None
        self.best_model_score = float("-inf")

    def train(self, X_train, pipelines, param_grids, scoring="silhouette_score", cv=5):
        dummy_y = np.zeros(X_train.shape[0])

        if scoring == "silhouette_score":
            scorer = silhouette_scorer
        elif scoring == "davies_bouldin_score":
            scorer = davies_bouldin_scorer
        elif scoring == "calinski_harabasz_score":
            scorer = calinski_harabasz_scorer
        else:
            raise ValueError(f"Unsupported scoring method: {scoring}")

        for model_name, pipeline in pipelines.items():
            wrapped_pipeline = ClusterEstimatorWrapper(pipeline)
            grid_search = GridSearchCV(
                wrapped_pipeline, param_grids[model_name], cv=cv, scoring=scorer
            )
            grid_search.fit(X_train, dummy_y)

            self.best_models[model_name] = grid_search.best_estimator_.estimator
            self.best_params[model_name] = grid_search.best_params_
            self.best_scores[model_name] = grid_search.best_score_

            if grid_search.best_score_ > self.best_model_score:
                self.best_model_score = grid_search.best_score_
                self.best_model_name = model_name

    def predict(self, X_test):
        best_model = self.best_models[self.best_model_name]
        return best_model.predict(X_test)
