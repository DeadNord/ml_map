import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    r2_score,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
    auc,
    confusion_matrix,
    classification_report,
)
from IPython.display import display
from sklearn import set_config
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import seaborn as sns


class SLModelEvaluator:
    """
    A class to evaluate and display model performance results.

    Methods
    -------
    display_results(X_valid, y_valid, best_models, best_params, best_scores, best_model_name, help_text=False):
        Displays the best parameters and evaluation metrics.
    validate_on_test(X_test, y_test, best_model, best_model_name):
        Validates the best model on the test set and displays evaluation metrics.
    visualize_pipeline(model_name, best_models):
        Visualizes the pipeline structure for a given model.
    feature_importance(X_train, y_train, df_original):
        Displays the feature importances using a RandomForest model.
    plot_roc_curve():
        Plots the ROC curve and displays the AUC score.
    plot_confusion_matrix():
        Plots the confusion matrix.
    """

    def __init__(self, model, X_test, y_test):
        """
        Инициализация класса SLModelEvaluator.

        Параметры:
        model: обученная модель
        X_test: тестовые данные
        y_test: истинные значения для тестовых данных
        """
        self.model = model
        self.X_test = X_test
        self.y_test = y_test
        self.y_pred = model.predict(X_test)
        if hasattr(model, "predict_proba"):
            self.y_pred_proba = model.predict_proba(X_test)[:, 1]
        else:
            self.y_pred_proba = None

    def display_results(
        self,
        X_valid,
        y_valid,
        best_models,
        best_params,
        best_scores,
        best_model_name,
        help_text=False,
    ):
        """
        Displays the best parameters and evaluation metrics.

        Parameters
        ----------
        X_valid : pd.DataFrame
            Validation data.
        y_valid : pd.Series
            Target values for the validation data.
        best_models : dict
            Dictionary of best models found by grid search for each model.
        best_params : dict
            Dictionary of best hyperparameters found by grid search for each model.
        best_scores : dict
            Dictionary of best scores achieved by grid search for each model.
        best_model_name : str
            Name of the best model based on the evaluation score.
        help_text : bool, optional
            Whether to display help text explaining the metrics (default is False).
        """
        results = []
        for model_name, model in best_models.items():
            y_pred = model.predict(X_valid)
            if hasattr(model, "predict_proba"):
                score = accuracy_score(y_valid, y_pred)
                f1 = f1_score(y_valid, y_pred, average="weighted")
                precision = precision_score(y_valid, y_pred, average="weighted")
                recall = recall_score(y_valid, y_pred, average="weighted")
                results.append(
                    {
                        "Model": model_name,
                        "Accuracy": score,
                        "F1 Score": f1,
                        "Precision": precision,
                        "Recall": recall,
                    }
                )
            else:
                mae = mean_absolute_error(y_valid, y_pred)
                mape = mean_absolute_percentage_error(y_valid, y_pred)
                r2 = r2_score(y_valid, y_pred)
                results.append(
                    {
                        "Model": model_name,
                        "R²": r2,
                        "MAE": mae,
                        "MAPE": mape,
                    }
                )

        results_df = pd.DataFrame(results).sort_values(
            by=list(results[0].keys())[1], ascending=False
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
            if hasattr(best_models[best_model_name], "predict_proba"):
                print(
                    "Accuracy: The ratio of correctly predicted instances to the total instances."
                )
                print("  - Range: [0, 1], higher is better.")
                print("  - Higher values indicate better model performance.")
                print("F1 Score: The harmonic mean of precision and recall.")
                print("  - Range: [0, 1], higher is better.")
                print("  - Higher values indicate better model performance.")
                print(
                    "Precision: The ratio of correctly predicted positive observations to the total predicted positives."
                )
                print("  - Range: [0, 1], higher is better.")
                print("  - Higher values indicate better model performance.")
                print(
                    "Recall: The ratio of correctly predicted positive observations to the all observations in actual class."
                )
                print("  - Range: [0, 1], higher is better.")
                print("  - Higher values indicate better model performance.")
            else:
                print(
                    "R²: The proportion of the variance in the dependent variable that is predictable from the independent variables."
                )
                print("  - Range: [0, 1], higher is better.")
                print("  - Higher values indicate better model performance.")
                print(
                    "MAE: The average of the absolute errors between the predicted and actual values."
                )
                print("  - Range: [0, ∞), lower is better.")
                print("  - Lower values indicate better model performance.")
                print(
                    "MAPE: The mean of the absolute percentage errors between the predicted and actual values."
                )
                print("  - Range: [0, ∞), lower is better.")
                print("  - Lower values indicate better model performance.")

    def validate_on_test(self, X_test, y_test, best_model, best_model_name):
        """
        Validates the best model on the test set and displays evaluation metrics.

        Parameters
        ----------
        X_test : pd.DataFrame
            Test data.
        y_test : pd.Series
            Target values for the test data.
        best_model : model
            Best model found by grid search.
        best_model_name : str
            Name of the best model.
        """
        y_pred = best_model.predict(X_test)
        if hasattr(best_model, "predict_proba"):
            score = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average="weighted")
            precision = precision_score(y_test, y_pred, average="weighted")
            recall = recall_score(y_test, y_pred, average="weighted")
            evaluation_df = pd.DataFrame(
                {
                    "Accuracy": [score],
                    "F1 Score": [f1],
                    "Precision": [precision],
                    "Recall": [recall],
                },
                index=[best_model_name],
            )
        else:
            mae = mean_absolute_error(y_test, y_pred)
            mape = mean_absolute_percentage_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            evaluation_df = pd.DataFrame(
                {"R²": [r2], "MAE": [mae], "MAPE": [f"{mape:.2%}"]},
                index=[best_model_name],
            )

        print(f"Results for {best_model_name}:")
        display(evaluation_df)

    def visualize_pipeline(self, model_name, best_models):
        """
        Visualizes the pipeline structure for a given model.

        Parameters
        ----------
        model_name : str
            Name of the model to visualize.
        best_models : dict
            Dictionary of best models found by grid search for each model.
        """
        set_config(display="diagram")
        return best_models[model_name]

    def feature_importance(self, X_train, y_train, df_original):
        """
        Displays the feature importances using a RandomForest model.

        Parameters
        ----------
        X_train : pd.DataFrame
            Training data.
        y_train : pd.Series
            Target values for the training data.
        df_original : pd.DataFrame
            Original dataframe with feature names.
        """
        feature_names = df_original.columns

        forest = (
            RandomForestClassifier(n_estimators=100, random_state=42)
            if y_train.nunique() > 2
            else RandomForestRegressor(n_estimators=100, random_state=42)
        )
        forest.fit(X_train, y_train)

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
