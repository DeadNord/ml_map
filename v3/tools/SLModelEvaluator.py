import pandas as pd
import torch
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
    auc,
    balanced_accuracy_score,
)
from IPython.display import display
import matplotlib.pyplot as plt
from sklearn.ensemble import (
    RandomForestRegressor,
    RandomForestClassifier,
    GradientBoostingRegressor,
    GradientBoostingClassifier,
)
from torchsummary import summary


class SLModelEvaluator:
    """
    A class to evaluate and display PyTorch model performance results.

    Methods
    -------
    display_results(X_valid, y_valid, model, task_type='regression', help_text=False):
        Displays the evaluation metrics for the given model.
    plot_roc_curve(X_test, y_test, model):
        Plots the ROC curve for classification tasks.
    plot_confusion_matrix(X_test, y_test, model):
        Plots the confusion matrix for classification tasks.
    validate_on_test(X_test, y_test, model, task_type):
        Validates the model on test data and displays metrics.
    feature_importance(X_train, y_train, df_original, model_type="random_forest", print_zero_importance=False):
        Displays the feature importances using RandomForest or similar model.
    """

    def display_results(
        self,
        X_valid,
        y_valid,
        best_models,
        best_params,
        best_scores,
        best_model_name,
        task_type="regression",
        help_text=False,
    ):
        """
        Displays the evaluation metrics for the best models and best parameters.
        """
        results = []

        for model_name, model_pipeline in best_models.items():
            if "regressor" in model_pipeline.named_steps:
                pytorch_regressor = model_pipeline.named_steps["regressor"]
                pytorch_model = pytorch_regressor.model
                device = pytorch_regressor.device
                print(f"Extracted PyTorch model: {pytorch_model}")
                print(f"Model is on device: {device}")

                pytorch_model.eval()

                with torch.no_grad():
                    X_tensor = torch.tensor(X_valid.to_numpy(), dtype=torch.float32).to(
                        device
                    )
                    y_pred = pytorch_model(X_tensor).cpu().numpy().flatten()

                if task_type == "regression":
                    mae = mean_absolute_error(y_valid, y_pred)
                    mape = mean_absolute_percentage_error(y_valid, y_pred)
                    mse = mean_squared_error(y_valid, y_pred)
                    r2 = r2_score(y_valid, y_pred)

                    results.append(
                        {
                            "Model": model_name,
                            "R²": r2,
                            "MAE": mae,
                            "MAPE": mape,
                            "MSE": mse,
                        }
                    )
                elif task_type == "classification":
                    y_pred_classes = (y_pred > 0.5).astype(int)
                    accuracy = accuracy_score(y_valid, y_pred_classes)
                    balanced_acc = balanced_accuracy_score(y_valid, y_pred_classes)
                    f1 = f1_score(y_valid, y_pred_classes)
                    precision = precision_score(y_valid, y_pred_classes)
                    recall = recall_score(y_valid, y_pred_classes)

                    results.append(
                        {
                            "Model": model_name,
                            "Accuracy": accuracy,
                            "Balanced Accuracy": balanced_acc,
                            "F1 Score": f1,
                            "Precision": precision,
                            "Recall": recall,
                        }
                    )
            else:
                print(f"Model {model_name} does not contain a 'regressor' step")

        results_df = pd.DataFrame(results).sort_values(
            by=list(results[0].keys())[1], ascending=False
        )
        param_df = (
            pd.DataFrame(best_params).T.reset_index().rename(columns={"index": "Model"})
        )

        best_model_df = pd.DataFrame(
            {
                "Overall Best Model": [best_model_name],
                "Score (based on cross-validation score)": [
                    best_scores[best_model_name]
                ],
            }
        )

        print("Evaluation Metrics for Validation Set:")
        display(results_df)

        print("\nBest Parameters for Each Model (found during cross-validation):")
        display(param_df)

        print("\nOverall Best Model and Score (based on cross-validation score):")
        display(best_model_df)

        if help_text:
            if task_type == "classification":
                print("\nMetric Explanations for Classification:")
                print(
                    "Accuracy: The ratio of correctly predicted instances to the total instances."
                )
                print(
                    "Balanced Accuracy: The average of recall obtained on each class."
                )
                print("F1 Score: Harmonic mean of precision and recall.")
                print(
                    "Precision: Ratio of correctly predicted positive observations to all positive predictions."
                )
                print(
                    "Recall: Ratio of correctly predicted positive observations to all actual positives."
                )
            elif task_type == "regression":
                print("\nMetric Explanations for Regression:")
                print(
                    "R²: Proportion of the variance explained by the model (higher is better)."
                )
                print(
                    "MAE: Mean Absolute Error, average error magnitude (lower is better)."
                )
                print("MAPE: Mean Absolute Percentage Error (lower is better).")
                print("MSE: Mean Squared Error (lower is better).")

    def plot_roc_curve(self, X_test, y_test, model):
        """
        Plots the ROC curve for classification tasks.

        Parameters
        ----------
        X_test : pd.DataFrame
            Test data.
        y_test : pd.Series
            True labels for the test data.
        model : nn.Module
            Trained PyTorch model.
        """
        model.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X_test, dtype=torch.float32)
            y_pred_proba = model(X_tensor).numpy().flatten()

        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(8, 6))
        plt.plot(
            fpr,
            tpr,
            color="darkorange",
            lw=2,
            label=f"ROC curve (area = {roc_auc:.2f})",
        )
        plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Receiver Operating Characteristic")
        plt.legend(loc="lower right")
        plt.show()

    def plot_confusion_matrix(self, X_test, y_test, model):
        """
        Plots the confusion matrix for classification tasks.

        Parameters
        ----------
        X_test : pd.DataFrame
            Test data.
        y_test : pd.Series
            True labels for the test data.
        model : nn.Module
            Trained PyTorch model.
        """
        model.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X_test, dtype=torch.float32)
            y_pred = model(X_tensor).numpy().flatten()

        y_pred_classes = (y_pred > 0.5).astype(int)
        cm = confusion_matrix(y_test, y_pred_classes)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(cmap="Blues")
        plt.show()

    def visualize_pipeline(self, model_name, best_models):
        """
        Визуализирует структуру PyTorch модели внутри пайплайна.

        Parameters
        ----------
        model_name : str
            Имя модели для визуализации.
        best_models : dict
            Словарь с лучшими моделями, найденными с помощью поиска по сетке.
        """
        pipeline = best_models.get(model_name)
        if pipeline is None:
            raise ValueError(f"Модель с именем {model_name} не найдена в best_models.")

        pytorch_regressor = pipeline.named_steps.get("regressor")
        if pytorch_regressor is None:
            raise ValueError(f"В пайплайне модели {model_name} не найден регрессор.")

        model = pytorch_regressor.model
        if isinstance(model, torch.nn.Module):
            print(f"Визуализация архитектуры модели: {model_name}")
            input_size = pytorch_regressor.input_size
            summary(model, input_size=(1, input_size))
        else:
            raise ValueError(
                f"Модель {model_name} не является PyTorch nn.Module, а {type(model)}"
            )

    def validate_on_test(self, X_test, y_test, best_models, best_model_name, task_type):
        """
        Validates the model on test data and displays evaluation metrics.

        Parameters
        ----------
        X_test : pd.DataFrame
            Test data.
        y_test : pd.Series or np.array
            Target values for the test data.
        best_models : dict
            Dictionary of best models from GridSearchCV.
        best_model_name : str
            Name of the best model to validate.
        task_type : str
            Either 'regression' or 'classification'.
        """
        best_model_pipeline = best_models[best_model_name]
        pytorch_regressor = best_model_pipeline.named_steps["regressor"]
        model = pytorch_regressor.model
        device = pytorch_regressor.device

        model.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X_test.to_numpy(), dtype=torch.float32).to(device)
            y_pred = model(X_tensor).cpu().numpy().flatten()

        if task_type == "classification":
            y_pred_classes = (y_pred > 0.5).astype(int)
            accuracy = accuracy_score(y_test, y_pred_classes)
            balanced_acc = balanced_accuracy_score(y_test, y_pred_classes)
            f1 = f1_score(y_test, y_pred_classes)
            precision = precision_score(y_test, y_pred_classes)
            recall = recall_score(y_test, y_pred_classes)

            evaluation_df = pd.DataFrame(
                {
                    "Accuracy": [accuracy],
                    "Balanced Accuracy": [balanced_acc],
                    "F1 Score": [f1],
                    "Precision": [precision],
                    "Recall": [recall],
                },
                index=["Model Evaluation"],
            )
        else:
            mae = mean_absolute_error(y_test, y_pred)
            mape = mean_absolute_percentage_error(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            evaluation_df = pd.DataFrame(
                {"R²": [r2], "MAE": [mae], "MAPE": [mape], "MSE": [mse]},
                index=["Model Evaluation"],
            )

        print(f"Results for the model on the test set:")
        display(evaluation_df)

    def plot_loss_history(self, best_models, best_model_name):
        """
        Plots the training and validation loss history of the provided PyTorch model.

        Parameters
        ----------
        best_models : dict
            Dictionary of best models from GridSearchCV.
        best_model_name : str
            Name of the best model to plot the loss history.
        """
        best_model_pipeline = best_models[best_model_name]
        pytorch_regressor = best_model_pipeline.named_steps["regressor"]

        if hasattr(pytorch_regressor, "train_loss_history") and hasattr(
            pytorch_regressor, "val_loss_history"
        ):
            plt.plot(pytorch_regressor.train_loss_history, label="Training Loss")
            plt.plot(pytorch_regressor.val_loss_history, label="Validation Loss")
            plt.title("Training vs Validation Loss per Epoch")
            plt.xlabel("Epochs")
            plt.ylabel("Loss (MSE)")
            plt.legend()
            plt.show()
        else:
            print("The provided model does not have a loss history.")
