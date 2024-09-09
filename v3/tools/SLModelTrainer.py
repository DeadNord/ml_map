from sklearn.model_selection import GridSearchCV, ParameterGrid
from tqdm import tqdm


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
        pipelines,
        param_grids,
        X_val=None,
        y_val=None,
        scoring=None,
        cv=5,
        verbose=0,
        n_jobs=-1,
        error_score="raise",
        use_progress_bar=True,
    ):
        """
        Train the PyTorch models using grid search and cross-validation.

        Parameters
        ----------
        X_train : pd.DataFrame
            The training data.
        y_train : pd.Series
            The target values for the training data.
        pipelines : dict
            A dictionary of model pipelines.
        param_grids : dict
            A dictionary of hyperparameter grids for grid search.
        X_val : pd.DataFrame, optional
            The validation data (optional).
        y_val : pd.Series, optional
            The target values for the validation data (optional).
        scoring : str or None, optional
            Scoring metric for grid search. If None, a default metric is chosen based on the task type.
        cv : int, optional
            The number of cross-validation folds (default is 5).
        verbose : int, optional
            The verbosity level (default is 0).
        n_jobs : int, optional
            The number of jobs to run in parallel (default is -1).
        error_score : str, optional
            How to handle errors during fitting (default is "raise").
        use_progress_bar : bool, optional
            If True, a progress bar and callback are used (default is True).
        """
        if scoring is None:
            scoring = "accuracy" if self.task_type == "classification" else "r2"

        print(f"Task type: {self.task_type.capitalize()}")
        print(f"Using scoring metric: {scoring}")
        print(f"Training on device: {self.device}")

        # Calculate the total number of fits
        total_fits = sum(
            len(list(ParameterGrid(param_grids[model_name]))) * cv
            for model_name in pipelines
        )

        # Create a unified progress bar if requested
        if use_progress_bar:
            pbar = tqdm(total=total_fits, desc="Total Progress")
        else:
            pbar = None

        for model_name, pipeline in pipelines.items():
            pipeline.set_params(regressor__device=self.device)

            grid_search = GridSearchCV(
                pipeline,
                param_grids[model_name],
                cv=cv,
                scoring=scoring,
                verbose=verbose,
                n_jobs=n_jobs,
                error_score=error_score,
            )

            if use_progress_bar:
                # Callback function to update the progress bar after each fold
                def progress_bar_callback(train_loss, val_loss):
                    pbar.update(1)
                    pbar.set_postfix({"Train Loss": train_loss, "Val Loss": val_loss})

                # Pass the callback to fit_params for the regressor
                fit_params = {
                    "regressor__X_val": X_val,
                    "regressor__y_val": y_val,
                    "regressor__fold_callback": progress_bar_callback,
                }
            else:
                fit_params = {
                    "regressor__X_val": X_val,
                    "regressor__y_val": y_val,
                }

            grid_search.fit(X_train, y_train, **fit_params)

            self.best_estimators[model_name] = grid_search.best_estimator_
            self.best_params[model_name] = grid_search.best_params_
            self.best_scores[model_name] = -grid_search.best_score_

            if self.best_scores[model_name] > self.best_model_score:
                self.best_model_name = model_name
                self.best_model_score = self.best_scores[model_name]

        if use_progress_bar:
            pbar.close()

    def help(self):
        """
        Provides information on how to use the SLModelTrainer class.
        """
        print("=== SLModelTrainer Help ===")
        print("This trainer supports both regression and classification tasks.")
        print("\nUsage:")
        print(
            "1. Initialize the SLModelTrainer with the task type ('regression' or 'classification') and the device ('cpu' or 'cuda')."
        )
        print("2. Create a pipeline with preprocessing and the PyTorchRegressor.")
        print("3. Define the parameter grid for hyperparameter search.")
        print(
            "4. Call the `train` method with the training data, pipeline, and parameter grid."
        )
        print("\nParameters:")
        print("- X_train : Training data (pandas DataFrame).")
        print("- y_train : Target values for the training data (pandas Series).")
        print("- X_val : Validation data (optional, pandas DataFrame).")
        print(
            "- y_val : Target values for the validation data (optional, pandas Series)."
        )
        print("- pipelines : Dictionary of model pipelines.")
        print("- param_grids : Dictionary of hyperparameter grids for GridSearchCV.")
        print(
            "- scoring : Metric for evaluating models (optional, default depends on task type)."
        )
        print("- cv : Number of cross-validation folds (default is 5).")
        print("- n_jobs : Number of jobs to run in parallel (default is -1).")
        print(
            "- use_progress_bar : If True, a progress bar and callbacks will be used during training."
        )
