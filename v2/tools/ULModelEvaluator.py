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
        cluster_labels_dict,
        best_models,
        best_params,
        best_scores,
        best_model_name,
        help_text=False,
    ):

        results = []
        for model_name, model in best_models.items():
            cluster_labels = cluster_labels_dict[model_name]
            silhouette = silhouette_score(X_train, cluster_labels)
            davies_bouldin = davies_bouldin_score(X_train, cluster_labels)
            calinski_harabasz = calinski_harabasz_score(X_train, cluster_labels)
            results.append(
                {
                    "Model": model_name,
                    "Silhouette Score": silhouette,
                    "Davies-Bouldin Index": davies_bouldin,
                    "Calinski-Harabasz Index": calinski_harabasz,
                }
            )

        results_df = pd.DataFrame(results).sort_values(
            by="Silhouette Score", ascending=False
        )
        param_df = (
            pd.DataFrame(best_params).T.reset_index().rename(columns={"index": "Model"})
        )

        print("Evaluation Metrics for Best Models:")
        display(results_df)

        print("\nBest Parameters for Each Model:")
        display(param_df)

        print(
            f"\nOverall Best Model: {best_model_name}, Score: {best_scores[best_model_name]}"
        )

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

    def visualize_pipeline(self, model_name, best_models):

        set_config(display="diagram")
        return best_models[model_name]

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
