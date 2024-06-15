import re
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    r2_score,
)
from sklearn import set_config
from IPython.display import display

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
    feature_importances : dict
        Dictionary of feature importances for each model.

    Methods
    -------
    train(X_train, y_train, pipelines, param_grids, scoring='neg_mean_absolute_percentage_error', cv=5):
        Trains the pipelines with grid search.

    evaluate(X_valid, y_valid):
        Evaluates the best models on the validation set.

    display_results(X_valid, y_valid):
        Displays the best parameters and evaluation metrics.

    validate_on_test(X_test, y_test):
        Validates the best model on the test set and displays evaluation metrics.

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
        self.feature_importances = {}

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

            # Extract regressor from the pipeline and save feature importances
            regressor = grid_search.best_estimator_.named_steps["regressor"]
            self.feature_importances[model_name] = {
                re.sub(r"^[^_]+__", "", feature): importance
                for feature, importance in zip(
                    X_train.columns, regressor.feature_importances_
                )
            }

            # Update the best model name based on the score
            if (
                self.best_model_name is None
                or self.best_scores[model_name] < self.best_scores[self.best_model_name]
            ):
                self.best_model_name = model_name

    def evaluate(self, X_valid, y_valid):
        """
        Evaluates the best models on the validation set.

        Parameters
        ----------
        X_valid : pd.DataFrame
            Validation data.
        y_valid : pd.Series
            Target values for the validation data.

        Returns
        -------
        dict
            Evaluation metrics including R², MAE, and MAPE for each model.
        """
        evaluation_metrics = {}
        for model_name, model in self.best_models.items():
            y_pred = model.predict(X_valid)
            mape = mean_absolute_percentage_error(y_valid, y_pred)
            mae = mean_absolute_error(y_valid, y_pred)
            r2 = r2_score(y_valid, y_pred)
            evaluation_metrics[model_name] = {"R²": r2, "MAE": mae, "MAPE": mape}
        return evaluation_metrics

    def display_results(self, X_valid, y_valid):
        """
        Displays the best parameters and evaluation metrics.

        Parameters
        ----------
        X_valid : pd.DataFrame
            Validation data.
        y_valid : pd.Series
            Target values for the validation data.
        """
        for model_name in self.best_models.keys():
            # Display best hyperparameters
            best_params = pd.DataFrame.from_dict(
                self.best_params[model_name], orient="index", columns=["Value"]
            )
            best_params["Metric"] = "Best Parameters"
            best_params.reset_index(inplace=True)
            best_params.rename(columns={"index": "Parameter"}, inplace=True)
            best_params = best_params[["Metric", "Parameter", "Value"]]

            print(f"Results for {model_name}:")
            display(best_params)

            # Display evaluation metrics
            evaluation_metrics = self.evaluate(X_valid, y_valid)[model_name]
            evaluation_df = pd.DataFrame(
                {
                    "Metric": ["R²", "MAE", "MAPE"],
                    "Value": [
                        evaluation_metrics["R²"],
                        evaluation_metrics["MAE"],
                        evaluation_metrics["MAPE"],
                    ],
                }
            )
            display(evaluation_df)

    def validate_on_test(self, X_test, y_test):
        """
        Validates the best model on the test set and displays evaluation metrics.

        Parameters
        ----------
        X_test : pd.DataFrame
            Test data.
        y_test : pd.Series
            Target values for the test data.
        """
        best_model = self.best_models[self.best_model_name]
        y_pred = best_model.predict(X_test)
        mape = mean_absolute_percentage_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        evaluation_df = pd.DataFrame(
            {"R²": [r2], "MAE": [mae], "MAPE": [f"{mape:.2%}"]},
            index=[self.best_model_name],
        )

        print(f"Results for {self.best_model_name}:")
        display(evaluation_df)

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


# Example usage:
# if __name__ == "__main__":
#     # Load your dataset
#     df = pd.read_csv("path_to_your_dataset.csv")

#     # Assume the data is already preprocessed
#     X_train, X_test, y_train, y_test = train_test_split(
#         df.drop(columns=["target"]), df["target"], test_size=0.2, random_state=42
#     )

#     # Define pipelines for different models
#     pipelines = {
#         "RandomForest": Pipeline(
#             [
#                 ("regressor", RandomForestRegressor(random_state=42)),
#             ]
#         ),
#         "GradientBoosting": Pipeline(
#             [
#                 ("regressor", GradientBoostingRegressor(random_state=42)),
#             ]
#         ),
#     }

#     # Define hyperparameters for different models
#     param_grids = {
#         "RandomForest": {
#             "regressor__n_estimators": [50, 100],
#             "regressor__max_features": ["sqrt", "log2"],
#             "regressor__max_depth": [10, 20, None],
#         },
#         "GradientBoosting": {
#             "regressor__n_estimators": [50, 100],
#             "regressor__learning_rate": [0.01, 0.1, 0.2],
#             "regressor__max_depth": [3, 5, 7],
#         },
#     }

#     # Initialize and train model pipeline
#     model_pipeline = ModelPipeline()
#     model_pipeline.train(X_train, y_train, pipelines, param_grids)

#     # Display results on validation set
#     model_pipeline.display_results(X_test, y_test)

#     # Validate on test set
#     model_pipeline.validate_on_test(X_test, y_test)

#     # Visualize pipeline for RandomForest
#     model_pipeline.visualize_pipeline("RandomForest")

#     # Visualize pipeline for GradientBoosting
#     model_pipeline.visualize_pipeline("GradientBoosting")
