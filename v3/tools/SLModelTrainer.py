from sklearn.model_selection import GridSearchCV
from tqdm import tqdm


class SLModelTrainer:
    """
    A class to train PyTorch models with grid search for hyperparameter tuning.
    Supports both regression and classification tasks.

    Methods
    -------
    train(X_train, y_train, X_val=None, y_val=None, pipelines, param_grids, scoring=None, cv=5, verbose=0, n_jobs=-1)
        Trains the PyTorch model with grid search for hyperparameters.
    help()
        Provides information on how to use the SLModelTrainer class.
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
    ):
        """
        Trains the PyTorch model with grid search for hyperparameters.

        Parameters
        ----------
        X_train : pd.DataFrame
            Training data.
        y_train : pd.Series
            Target values for the training data.
        X_val : pd.DataFrame, optional
            Validation data. Default is None.
        y_val : pd.Series, optional
            Target values for the validation data. Default is None.
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
        error_score : str, optional
            How to handle errors during fitting. Default is 'raise'.
        """
        if scoring is None:
            scoring = "accuracy" if self.task_type == "classification" else "r2"

        print(f"Task type: {self.task_type.capitalize()}")
        print(f"Using scoring metric: {scoring}")
        print(f"Training on device: {self.device}")

        total_fits = 0

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

            # Calculate the total number of fits
            total_fits = len(param_grids[model_name]) * cv

            # tqdm progress bar
            with tqdm(total=total_fits, desc=f"Fitting {model_name}") as pbar:
                # Callback function to update the progress bar
                def progress_bar_callback(*args, **kwargs):
                    pbar.update(1)

                grid_search.fit(
                    X_train,
                    y_train,
                    regressor__X_val=X_val,
                    regressor__y_val=y_val,
                    callbacks=[progress_bar_callback],  # Enabling the callback
                )

            self.best_estimators[model_name] = grid_search.best_estimator_
            self.best_params[model_name] = grid_search.best_params_
            self.best_scores[model_name] = -grid_search.best_score_

            if self.best_scores[model_name] > self.best_model_score:
                self.best_model_name = model_name
                self.best_model_score = self.best_scores[model_name]

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
        print("\nNote:")
        print(
            "- If X_val and y_val are not provided, the model will train without validation."
        )
        print("- Make sure to pass compatible pipelines and parameter grids.")
