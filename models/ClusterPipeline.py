from itertools import product
from matplotlib import pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import silhouette_score
from sklearn import set_config
import pandas as pd
from IPython.display import display
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
)
from sklearn.base import clone


class ModelPipeline:
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
    train(X_train, pipelines, param_grids, scoring='silhouette_score', cv=5):
        Trains the pipelines with grid search.

    display_results():
        Displays the best parameters and evaluation metrics.

    visualize_pipeline(model_name):
        Visualizes the pipeline structure for a given model.
    """

    def __init__(self):
        """
        Constructs all the necessary attributes for the ModelPipeline object.
        """
        self.best_models = {}
        self.best_params = {}
        self.best_scores = {}
        self.best_model_name = None

    def train(self, X_train, pipelines, param_grids, scoring="silhouette_score"):
        for model_name, pipeline in pipelines.items():
            param_grid = param_grids[model_name]
            self.best_scores[model_name] = float("-inf")

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

                if score > self.best_scores[model_name]:
                    self.best_scores[model_name] = score
                    self.best_models[model_name] = model
                    self.best_params[model_name] = param_dict
                    self.best_model_name = model_name

    def display_results(self, X_train, help_text=False):
        """
        Displays the best parameters and evaluation metrics.

        Parameters
        ----------
        X_train : pd.DataFrame
            The training data.
        help_text : bool, optional
            Whether to display the help text for interpreting the metrics (default is False).
        """
        silhouette_scores = []
        davies_bouldin_scores = []
        calinski_harabasz_scores = []

        # Extract unique parameters
        param_keys = set()
        for params in self.best_params.values():
            param_keys.update(params.keys())

        # Prepare the data for parameters table
        param_data = {key: [] for key in param_keys}
        param_data["Model"] = []

        for model_name in self.best_models.keys():
            best_model = self.best_models[model_name]
            cluster_labels = best_model.predict(X_train)

            silhouette = silhouette_score(X_train, cluster_labels)
            davies_bouldin = davies_bouldin_score(X_train, cluster_labels)
            calinski_harabasz = calinski_harabasz_score(X_train, cluster_labels)

            silhouette_scores.append(silhouette)
            davies_bouldin_scores.append(davies_bouldin)
            calinski_harabasz_scores.append(calinski_harabasz)

            # Add model name and parameters to the param_data
            param_data["Model"].append(model_name)
            for key in param_keys:
                param_data[key].append(self.best_params[model_name].get(key, None))

        results_df = pd.DataFrame(
            {
                "Model": list(self.best_models.keys()),
                "Silhouette Score": silhouette_scores,
                "Davies-Bouldin Index": davies_bouldin_scores,
                "Calinski-Harabasz Index": calinski_harabasz_scores,
            }
        )

        # Create DataFrame for parameters and ensure "Model" is the first column
        param_df = pd.DataFrame(param_data)
        param_df = param_df[
            ["Model"] + [col for col in param_df.columns if col != "Model"]
        ]

        print("Evaluation Metrics for Best Models:")
        display(results_df)

        print("\nBest Parameters for Each Model:")
        display(param_df)

        if help_text:
            print("\nMetric Explanations:")
            print(
                "Silhouette Score: Measures how similar an object is to its own cluster compared to other clusters."
            )
            print("  - Range: [-1, 1], higher is better.")
            print("  - Higher values indicate better-defined clusters.")
            print(
                "Davies-Bouldin Index: Measures the average similarity ratio of each cluster with its most similar cluster."
            )
            print("  - Range: [0, ∞), lower is better.")
            print("  - Lower values indicate better clustering.")
            print(
                "Calinski-Harabasz Index: Ratio of the sum of between-cluster dispersion to within-cluster dispersion."
            )
            print("  - Range: [0, ∞), higher is better.")
            print("  - Higher values indicate better-defined clusters.")

    def visualize_pipeline(self, model_name):
        """
        Visualizes the pipeline structure for a given model.

        Parameters
        ----------
        model_name : str
            Name of the model to visualize.
        """
        set_config(display="diagram")
        return self.best_models[model_name]

    def generate_cluster_report(self, df_original, df_transformed):
        """
        Generates a report with descriptive statistics for each cluster.

        Parameters
        ----------
        df_original : pd.DataFrame
            The original dataset.
        df_transformed : pd.DataFrame
            The transformed dataset used for clustering.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing median values for each feature and the count of objects in each cluster.
        """
        best_kmeans = self.best_models[self.best_model_name]
        df_original["Cluster"] = best_kmeans.predict(df_transformed)

        # Calculate medians for each cluster
        cluster_report = df_original.groupby("Cluster").median()

        # Add count of objects in each cluster
        cluster_report["ObjectCount"] = df_original["Cluster"].value_counts()

        return cluster_report

    def feature_importance(self, X_train, df_original):
        """
        Evaluates and visualizes feature importance using Random Forest.

        Parameters
        ----------
        X_train : np.ndarray
            The training data.
        df_original : pd.DataFrame
            The original DataFrame to get feature names.
        """
        if not isinstance(X_train, np.ndarray):
            raise ValueError("X_train must be a numpy array")
        if not isinstance(df_original, pd.DataFrame):
            raise ValueError("df_original must be a pandas DataFrame")

        feature_names = df_original.columns

        best_kmeans = self.best_models[self.best_model_name]
        cluster_labels = best_kmeans.predict(X_train)

        forest = RandomForestClassifier(n_estimators=100, random_state=42)
        forest.fit(X_train, cluster_labels)

        importances = forest.feature_importances_
        indices = np.argsort(importances)[::-1]

        sorted_importances = importances[indices]
        sorted_features = feature_names[indices]

        plt.figure(figsize=(10, 6))
        plt.title("Feature Importance")
        plt.barh(range(len(indices)), sorted_importances, align="center")
        plt.yticks(range(len(indices)), sorted_features)
        plt.xlabel("Relative Importance")
        plt.gca().invert_yaxis()
        plt.show()
