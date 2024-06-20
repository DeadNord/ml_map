import numpy as np
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
)
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
from sklearn import set_config
from sklearn.ensemble import RandomForestClassifier


class ULModelEvaluator:
    def display_results(
        self,
        X_train,
        cluster_labels,
        best_model,
        best_params,
        best_score,
        help_text=False,
    ):

        silhouette = silhouette_score(X_train, cluster_labels)
        davies_bouldin = davies_bouldin_score(X_train, cluster_labels)
        calinski_harabasz = calinski_harabasz_score(X_train, cluster_labels)

        results_df = pd.DataFrame(
            {
                "Model": [best_model.__class__.__name__],
                "Silhouette Score": [silhouette],
                "Davies-Bouldin Index": [davies_bouldin],
                "Calinski-Harabasz Index": [calinski_harabasz],
            }
        )

        param_df = pd.DataFrame(best_params, index=[0])

        print("Evaluation Metrics for Best Model:")
        display(results_df)

        print("\nBest Parameters for the Model:")
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

    def visualize_pipeline(self, best_model):

        set_config(display="diagram")
        return best_model

    def generate_cluster_report(self, df_original, cluster_labels):
        df_original["Cluster"] = cluster_labels

        cluster_report = df_original.groupby("Cluster").median()
        cluster_report["ObjectCount"] = df_original["Cluster"].value_counts()

        return cluster_report

    def feature_importance(self, X_train, cluster_labels, df_original):

        feature_names = df_original.columns

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
