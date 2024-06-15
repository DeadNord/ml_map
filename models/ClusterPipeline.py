from sklearn.model_selection import GridSearchCV
from sklearn.metrics import silhouette_score
from sklearn import set_config


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

    Methods
    -------
    train(X_train, pipelines, param_grids, scoring='silhouette_score', cv=5):
        Trains the pipelines with grid search.

    display_results():
        Displays the best parameters and evaluation metrics.

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

    def train(
        self,
        X_train,
        pipelines,
        param_grids,
        scoring="silhouette_score",
        cv=5,
    ):
        """
        Trains the pipelines with grid search.

        Parameters
        ----------
        X_train : pd.DataFrame
            Training data.
        pipelines : dict
            Dictionary of model pipelines.
        param_grids : dict
            Dictionary of hyperparameter grids for grid search.
        scoring : str, optional
            Scoring metric for grid search (default is 'silhouette_score').
        cv : int, optional
            Number of cross-validation folds (default is 5).
        """
        for model_name, pipeline in pipelines.items():
            grid_search = GridSearchCV(
                pipeline,
                param_grids[model_name],
                cv=cv,
                scoring=(
                    self._silhouette_scorer
                    if scoring == "silhouette_score"
                    else scoring
                ),
            )

            grid_search.fit(X_train)

            self.best_models[model_name] = grid_search.best_estimator_
            self.best_params[model_name] = grid_search.best_params_
            self.best_scores[model_name] = grid_search.best_score_

            # Update the best model name based on the score
            if (
                self.best_model_name is None
                or self.best_scores[model_name] > self.best_scores[self.best_model_name]
            ):
                self.best_model_name = model_name

    def _silhouette_scorer(self, estimator, X):
        cluster_labels = estimator.fit_predict(X)
        return silhouette_score(X, cluster_labels)

    def display_results(self):
        """
        Displays the best parameters and evaluation metrics.
        """
        print(f"Best model: {self.best_model_name}")
        print(f"Best parameters: {self.best_params[self.best_model_name]}")
        print(f"Best silhouette score: {self.best_scores[self.best_model_name]}")

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


# # Пример использования:
# from sklearn.preprocessing import StandardScaler
# from sklearn.pipeline import Pipeline
# from sklearn.cluster import KMeans

# # Пример данных
# data = {
#     "Cement": [540.0, 540.0, 332.5, 332.5, 198.6],
#     "BlastFurnaceSlag": [0.0, 0.0, 142.5, 142.5, 132.4],
#     "FlyAsh": [0.0, 0.0, 0.0, 0.0, 0.0],
#     "Water": [162.0, 162.0, 228.0, 228.0, 192.0],
#     "Superplasticizer": [2.5, 2.5, 0.0, 0.0, 0.0],
#     "CoarseAggregate": [1040.0, 1055.0, 932.0, 932.0, 978.4],
#     "FineAggregate": [676.0, 676.0, 594.0, 594.0, 825.5],
#     "Age": [28, 28, 270, 365, 360],
#     "CompressiveStrength": [79.99, 61.89, 40.27, 41.05, 44.30],
# }

# df = pd.DataFrame(data)

# # Нормализация данных
# scaler = StandardScaler()
# df_scaled = scaler.fit_transform(df)

# # Определение pipeline для KMeans
# pipelines = {
#     "KMeans": Pipeline(
#         [
#             ("cluster", KMeans(random_state=42)),
#         ]
#     ),
# }

# # Определение гиперпараметров для KMeans
# param_grids = {
#     "KMeans": {
#         "cluster__n_clusters": [2, 3, 4, 5, 6, 7, 8, 9, 10],
#     },
# }

# # Инициализация и обучение ModelPipeline
# model_pipeline = ModelPipeline()
# model_pipeline.train(df_scaled, pipelines, param_grids)

# # Отображение результатов
# model_pipeline.display_results()

# # Получение меток кластеров
# best_kmeans = model_pipeline.best_models["KMeans"]
# df["Cluster"] = best_kmeans.predict(df_scaled)

# # Вывод результатов
# print(df)
