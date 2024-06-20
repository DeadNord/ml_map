from itertools import product
from sklearn.base import clone
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
)


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

    def train(self, X_train, X_val, pipelines, param_grids, scoring="silhouette_score"):
        for model_name, pipeline in pipelines.items():
            param_grid = param_grids[model_name]

            param_combinations = list(product(*param_grid.values()))
            self.best_scores[model_name] = float("-inf")

            for params in param_combinations:
                param_dict = {
                    param_name: param_value
                    for param_name, param_value in zip(param_grid.keys(), params)
                }
                model = clone(pipeline)
                model.set_params(**param_dict)
                model.fit(X_train)

                # Use fit_predict if the model has no predict method
                if hasattr(model, "predict"):
                    cluster_labels = model.predict(X_train)
                else:
                    cluster_labels = model.fit_predict(X_train)

                # Check if the number of unique labels is valid
                if len(set(cluster_labels)) < 2:
                    continue

                if scoring == "silhouette_score":
                    score = silhouette_score(X_val, cluster_labels)
                elif scoring == "davies_bouldin_score":
                    score = -davies_bouldin_score(X_val, cluster_labels)
                elif scoring == "calinski_harabasz_score":
                    score = calinski_harabasz_score(X_val, cluster_labels)
                else:
                    raise ValueError(f"Unsupported scoring method: {scoring}")

                if score > self.best_scores[model_name]:
                    self.best_scores[model_name] = score
                    self.best_models[model_name] = model
                    self.best_params[model_name] = param_dict

                if score > self.best_model_score:
                    self.best_model_score = score
                    self.best_model_name = model_name

    # def predict(self, X_test):
    #     best_model = self.best_models[self.best_model_name]
    #     # Use fit_predict if the model has no predict method
    #     if hasattr(best_model, "predict"):
    #         return best_model.predict(X_test)
    #     else:
    #         return best_model.fit_predict(X_test)
