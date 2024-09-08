from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    mean_absolute_error,
    mean_absolute_percentage_error,
    r2_score,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
    auc,
    balanced_accuracy_score,
)


class SLModelTrainer:
    """
    A class to train PyTorch models with grid search for hyperparameter tuning.
    Supports both regression and classification tasks.
    """

    def __init__(self, task_type, device="cpu"):
        """
        Initialize the trainer with a specific task type and device.

        Parameters
        ----------
        task_type : str
            The type of task ('regression' or 'classification').
        device : str, optional
            The device to train the model on ('cpu' or 'cuda'). Default is 'cpu'.
        """
        if task_type not in ["regression", "classification"]:
            raise ValueError(
                "Invalid task type. Choose 'regression' or 'classification'."
            )
        self.task_type = task_type
        self.device = device
        self.best_estimators = {}
        self.best_params = {}
        self.best_scores = {}
        self.best_model_name = None
        self.best_model_score = float("-inf")

    def train(
        self,
        X_train,
        y_train,
        X_val,
        y_val,
        pipelines,
        param_grids,
        scoring=None,
        cv=5,
        verbose=0,
        n_jobs=-1,
        error_score="raise",
    ):
        """
        Trains the PyTorch model with grid search for hyperparameters.

        Parameters
        ----------
        X_train : pd.DataFrame
            Training data.
        y_train : pd.Series
            Target values for the training data.
        X_val : pd.DataFrame
            Validation data.
        y_val : pd.Series
            Target values for the validation data.
        pipelines : dict
            Dictionary of model pipelines.
        param_grids : dict
            Dictionary of hyperparameter grids for grid search.
        scoring : str or None, optional
            Scoring metric for grid search. If None, a default metric is chosen based on the task type.
        cv : int, optional
            Number of cross-validation folds (default is 5).
        verbose : int, optional
            Verbosity level (default is 0).
        n_jobs : int, optional
            Number of jobs to run in parallel (default is -1).
        """
        if scoring is None:
            if self.task_type == "classification":
                scoring = "accuracy"
            else:
                scoring = "r2"

        print(f"Task type: {self.task_type.capitalize()}")
        print(f"Using scoring metric: {scoring}")
        print(f"Training on device: {self.device}")

        for model_name, pipeline in pipelines.items():
            pipeline.set_params(regressor__device=self.device)

            grid_search = GridSearchCV(
                pipeline,
                param_grids[model_name],
                cv=cv,
                scoring=scoring,
                verbose=verbose,
                n_jobs=n_jobs,
            )
            grid_search.fit(
                X_train, y_train, regressor__X_val=X_val, regressor__y_val=y_val
            )

            self.best_estimators[model_name] = grid_search.best_estimator_
            self.best_params[model_name] = grid_search.best_params_
            self.best_scores[model_name] = grid_search.best_score_

            if self.best_scores[model_name] > self.best_model_score:
                self.best_model_name = model_name
                self.best_model_score = self.best_scores[model_name]

    def help(self):
        """
        Prints help information about available metrics for classification and regression.
        """
        print("=== PyTorchModelTrainer Help ===")
        print("This trainer supports both regression and classification tasks.")
        print("\nAvailable scoring metrics:")
        print(
            "- For classification: accuracy, f1, precision, recall, balanced_accuracy"
        )
        print(
            "- For regression: r2, neg_mean_absolute_error, neg_mean_squared_error, neg_root_mean_squared_error"
        )
        print("\nUsage:")
        print("You can specify the scoring parameter when calling the train method.")
        print(
            "If you don't specify a scoring metric, the default will be 'accuracy' for classification and 'r2' for regression."
        )
        print(
            "You can also check available scoring metrics in the sklearn documentation."
        )
