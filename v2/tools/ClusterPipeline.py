from sklearn import set_config
from sklearn.base import clone
from itertools import product
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
)
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from IPython.display import display
import numpy as np


class ClusterPipeline:
    def __init__(self):
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

                # Перехват данных после feature_engineering
                X_train_transformed = model.named_steps[
                    "feature_engineering"
                ].fit_transform(X_train)
                print(
                    f"Data after feature engineering (model: {model_name}, params: {param_dict}):"
                )
                display(X_train_transformed.head())

                # Перехват данных после preprocessing
                X_train_preprocessed = model.named_steps["preprocessing"].fit_transform(
                    X_train_transformed
                )
                print(
                    f"Data after preprocessing (model: {model_name}, params: {param_dict}):"
                )
                display(X_train_preprocessed.head())

                model.named_steps["model"].fit(X_train_preprocessed)
                cluster_labels = model.named_steps["model"].predict(
                    X_train_preprocessed
                )

                if scoring == "silhouette_score":
                    score = silhouette_score(X_train_preprocessed, cluster_labels)
                elif scoring == "davies_bouldin_score":
                    score = -davies_bouldin_score(X_train_preprocessed, cluster_labels)
                elif scoring == "calinski_harabasz_score":
                    score = calinski_harabasz_score(
                        X_train_preprocessed, cluster_labels
                    )
                else:
                    raise ValueError(f"Unsupported scoring method: {scoring}")

                if score > self.best_scores[model_name]:
                    self.best_scores[model_name] = score
                    self.best_models[model_name] = model
                    self.best_params[model_name] = param_dict
                    self.best_model_name = model_name

    def display_results(self, X_train, help_text=False):
        silhouette_scores = []
        davies_bouldin_scores = []
        calinski_harabasz_scores = []

        param_keys = set()
        for params in self.best_params.values():
            param_keys.update(params.keys())

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
        set_config(display="diagram")
        return self.best_models[model_name]

    def generate_cluster_report(self, df_original):
        """
        Generates a report with descriptive statistics for each cluster.

        Parameters
        ----------
        df_original : pd.DataFrame
            The original dataset.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing median values for each feature and the count of objects in each cluster.
        """
        # Data transformation using preprocessing pipeline
        best_pipeline = self.best_models[self.best_model_name]
        feature_engineering_pipeline = best_pipeline.named_steps["feature_engineering"]
        preprocessor = best_pipeline.named_steps["preprocessing"]

        # Applying feature_engineering_pipeline
        df_transformed = feature_engineering_pipeline.transform(df_original)
        # Applying preprocessor
        df_transformed = preprocessor.fit_transform(df_transformed)

        # Getting cluster labels
        best_kmeans = best_pipeline.named_steps["model"]
        # cluster_labels = best_kmeans.predict(df_transformed)
        df_original["Cluster"] = best_kmeans.predict(df_transformed)
        # Calculate medians for each cluster
        cluster_report = df_original.groupby("Cluster").median()

        # Add count of objects in each cluster
        cluster_report["ObjectCount"] = df_original["Cluster"].value_counts()

        return cluster_report

    def feature_importance(self, df_original):
        if not isinstance(df_original, pd.DataFrame):
            raise ValueError("df_original must be a pandas DataFrame")

        # Data transformation
        best_pipeline = self.best_models[self.best_model_name]
        X_transformed = best_pipeline[:-1].transform(df_original)

        feature_names = best_pipeline[:-1].get_feature_names_out()
        best_kmeans = best_pipeline[-1]
        cluster_labels = best_kmeans.predict(X_transformed)

        forest = RandomForestClassifier(n_estimators=100, random_state=42)
        forest.fit(X_transformed, cluster_labels)

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
