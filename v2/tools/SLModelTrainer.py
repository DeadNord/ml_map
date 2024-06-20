from sklearn.model_selection import GridSearchCV


class SLModelTrainer:
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
    train(X_train, y_train, pipelines, param_grids, scoring='neg_mean_absolute_percentage_error', cv=5):
        Trains the pipelines with grid search.
    """

    def __init__(self):
        self.best_models = {}
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
        scoring="neg_mean_absolute_percentage_error",
        cv=5,
    ):
        """
        Trains the pipelines with grid search.

        Parameters
        ----------
        X_train : pd.DataFrame
            Training data.
        y_train : pd.Series
            Target values for the training data.
        pipelines : dict
            Dictionary of model pipelines.
        param_grids : dict
            Dictionary of hyperparameter grids for grid search.
        scoring : str, optional
            Scoring metric for grid search (default is 'neg_mean_absolute_percentage_error').
        cv : int, optional
            Number of cross-validation folds (default is 5).
        """
        for model_name, pipeline in pipelines.items():
            grid_search = GridSearchCV(
                pipeline, param_grids[model_name], cv=cv, scoring=scoring
            )
            grid_search.fit(X_train, y_train)

            self.best_models[model_name] = grid_search.best_estimator_
            self.best_params[model_name] = grid_search.best_params_
            self.best_scores[model_name] = -grid_search.best_score_

            # Update the best model name based on the score
            if (
                self.best_model_name is None
                or self.best_scores[model_name] < self.best_scores[self.best_model_name]
            ):
                self.best_model_name = model_name
