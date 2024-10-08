import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_regression
import numpy as np
import seaborn as sns
from IPython.display import display
from yellowbrick.cluster import KElbowVisualizer
from sklearn.cluster import KMeans


class EDA:
    """
    A class to perform Exploratory Data Analysis (EDA) on a dataset.

    Attributes
    ----------
    df : pd.DataFrame
        The dataset to be analyzed.
    """

    def __init__(self, df):
        """
        Constructs all the necessary attributes for the EDA object.

        Parameters
        ----------
        df : pd.DataFrame
            The dataset to be analyzed.
        """
        self.mutual_info_df = []
        if isinstance(df, pd.DataFrame):
            self.df = df
        else:
            raise ValueError("The input data must be a pandas DataFrame.")

    def missing_values_percentage(self, target=None):
        """
        Displays the percentage of missing values in each column along with additional statistics.

        Parameters
        ----------
        target : str, optional
            The target column to exclude from the calculation (default is None).
        """
        if target and target in self.df.columns:
            df = self.df.drop(columns=[target])
        else:
            df = self.df

        missing_ratios = df.isnull().mean() * 100
        missing_df = pd.DataFrame(
            {
                "Column": missing_ratios.index,
                "Missing Percentage": missing_ratios.values,
            }
        ).sort_values(by="Missing Percentage", ascending=False)

        avg_missing = missing_ratios.mean()
        median_missing = missing_ratios.median()
        max_missing = missing_ratios.max()
        min_missing = missing_ratios.min()
        std_missing = missing_ratios.std()
        q1_missing = missing_ratios.quantile(0.25)
        q3_missing = missing_ratios.quantile(0.75)
        iqr_missing = q3_missing - q1_missing

        summary_df = pd.DataFrame(
            {
                "Average Missing Percentage": [avg_missing],
                "Median Missing Percentage": [median_missing],
                "Maximum Missing Percentage": [max_missing],
                "Minimum Missing Percentage": [min_missing],
                "Standard Deviation of Missing Percentage": [std_missing],
                "First Quartile (Q1) of Missing Percentage": [q1_missing],
                "Third Quartile (Q3) of Missing Percentage": [q3_missing],
                "Interquartile Range (IQR) of Missing Percentage": [iqr_missing],
            }
        )

        print("Percentage of Missing Values in Each Column:")
        display(missing_df)
        print("\nSummary of Missing Values:")
        display(summary_df)

    def dataset_info(self):
        """
        Prints information about the dataset.
        """
        print("Dataset Information:\n")
        display(self.df.info())

    def dataset_custom_info(self):
        """
        Prints custom information about the dataset.
        """
        info_table = pd.DataFrame(
            {
                "Column": self.df.columns,
                "Has_Nulls": self.df.isnull().any(),
                "Dtype": self.df.dtypes,
            }
        )
        display(info_table)

    def dataset_shape(self):
        """
        Prints the shape of the dataset.
        """
        print("\nDataset Shape:\n")
        print(self.df.shape)

    def descriptive_statistics(self):
        """
        Displays descriptive statistics of the dataset.
        """
        print("\nDescriptive Statistics:\n")
        display(self.df.describe().transpose())

    def missing_values(self):
        """
        Displays the count of missing values in the dataset.
        """
        print("\nMissing Values:\n")
        display(self.df.isnull().sum())

    def sample_data(self, n=5):
        """
        Displays a sample of the dataset.

        Parameters
        ----------
        n : int, optional
            Number of rows to display (default is 5).
        """
        print("\nSample Data:\n")
        display(self.df.head(n))

    def plot_histogram(self, y=None):
        """
        Plots histograms for all numeric columns against the specified target column y.

        Parameters
        ----------
        y : str, optional
            The target column to plot histograms against all numeric columns (default is None).
        """
        numeric_columns = self.df.select_dtypes(include=["int64", "float64"]).columns

        if y and y in numeric_columns:
            for column in numeric_columns:
                if column != y:
                    plt.figure(figsize=(10, 6))
                    plt.hist(self.df[column], bins=30, edgecolor="black")
                    plt.title(f"Histogram of {column} vs {y}")
                    plt.xlabel(column)
                    plt.ylabel(y)
                    plt.show()
        else:
            for column in numeric_columns:
                plt.figure(figsize=(10, 6))
                plt.hist(self.df[column], bins=30, edgecolor="black")
                plt.title(f"Histogram of {column}")
                plt.xlabel(column)
                plt.ylabel("Frequency")
                plt.show()

    def plot_scatter(self, y=None):
        """
        Plots scatter plots for all pairs of numeric columns or against a specified target column.

        Parameters
        ----------
        y : str, optional
            The target column to plot against all numeric columns (default is None).
        """
        numeric_columns = self.df.select_dtypes(include=["int64", "float64"]).columns

        if y and y in numeric_columns:
            for column in numeric_columns:
                if column != y:
                    plt.figure(figsize=(10, 6))
                    plt.scatter(self.df[column], self.df[y], alpha=0.7)
                    plt.title(f"Scatter Plot of {column} vs {y}")
                    plt.xlabel(column)
                    plt.ylabel(y)
                    plt.show()
        else:
            for i, col_i in enumerate(numeric_columns):
                for j, col_j in enumerate(numeric_columns):
                    if i < j:
                        plt.figure(figsize=(10, 6))
                        plt.scatter(self.df[col_i], self.df[col_j], alpha=0.7)
                        plt.title(f"Scatter Plot of {col_i} vs {col_j}")
                        plt.xlabel(col_i)
                        plt.ylabel(col_j)
                        plt.show()

    def plot_target_distribution(self, target):
        """
        Plots the distribution of the target variable.

        Parameters
        ----------
        target : str
            The target column to plot the distribution for.
        """
        if target not in self.df.columns:
            raise ValueError(f"Target column '{target}' not found in DataFrame.")

        plt.figure(figsize=(10, 6))
        sns.histplot(self.df[target], bins=30, kde=True)
        plt.title(f"Distribution of {target}")
        plt.xlabel(target)
        plt.ylabel("Frequency")
        plt.show()

    def plot_correlation_matrix(self, method="pearson"):
        """
        Plots a correlation matrix for all numeric features in the dataset.

        Parameters
        ----------
        method : str, optional
            The method to compute the correlation ('pearson', 'kendall', 'spearman'), by default 'pearson'.
        """
        numeric_columns = self.df.select_dtypes(include=["int64", "float64"]).columns
        correlation_matrix = self.df[numeric_columns].corr(method=method)

        plt.figure(figsize=(12, 8))
        sns.heatmap(
            correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5
        )
        plt.title(f"Correlation Matrix ({method.title()})")
        plt.show()

    def calculate_mutual_information(self, target, discrete_features, plot=False):
        """
        Calculates mutual information between all features and the specified target column. Optionally plots the results.

        Parameters
        ----------
        target : str
            The target column to calculate mutual information against.
        discrete_features : list
            List of features to be considered as discrete.
        plot : bool, optional
            Whether to plot the mutual information scores (default is False).
        """
        if target not in self.df.columns:
            raise ValueError(f"Target column '{target}' not found in DataFrame.")

        features = self.df.drop(columns=[target], errors="ignore")
        target_values = self.df[target]

        # Convert categorical features to numerical
        for col in features.columns:
            if features[col].dtype == "object" or col in discrete_features:
                features[col], _ = features[col].factorize()

        # Calculate mutual information
        mi = mutual_info_regression(
            features,
            target_values,
            discrete_features=[
                features.columns.get_loc(c)
                for c in discrete_features
                if c in features.columns
            ],
        )
        mi_series = pd.Series(mi, index=features.columns, name="Mutual Information")
        mi_series = mi_series.sort_values(ascending=True)

        self.mutual_info_df = pd.DataFrame(
            {"Feature": features.columns, "MutualInfo": mi}
        )

        # Optionally plot the Mutual Information scores
        if plot:
            plt.figure(figsize=(6, 8))
            plt.barh(np.arange(len(mi_series)), mi_series)
            plt.yticks(np.arange(len(mi_series)), mi_series.index)
            plt.title("Mutual Information Scores")
            plt.xlabel("Mutual Information")
            plt.ylabel("Features")
            plt.show()
        else:
            print(mi_series.sort_values(ascending=False))

    def scale_and_plot_importance(
        self, feature_importance, rank_pct=True, sort_by=None
    ):
        """
        Scale/unify mutual information and feature importance, then plot the results.

        Parameters
        ----------
        feature_importance_df : pd.DataFrame
            DataFrame containing feature importance values.
        rank_pct : bool, optional
            Whether to scale values using rank percentage (default is True).
        sort_by : str, optional
            Column name to sort the DataFrame by (default is None).
        """
        # Convert feature_importance to DataFrame
        feature_importance_df = pd.DataFrame(
            list(feature_importance.items()), columns=["Feature", "FeatureImportance"]
        )

        # Merge mutual_info_df and feature_importance_df
        importance_df = pd.merge(
            self.mutual_info_df, feature_importance_df, on="Feature"
        )

        # Apply rank percentage if specified
        if rank_pct:
            importance_df["MutualInfoRank"] = importance_df["MutualInfo"].rank(pct=True)
            importance_df["FeatureImportanceRank"] = importance_df[
                "FeatureImportance"
            ].rank(pct=True)

        # Sort the DataFrame if specified
        if sort_by:
            importance_df = importance_df.sort_values(by=sort_by, ascending=False)

        # Plot the results
        self.plot_importance(importance_df)

        return importance_df

    def plot_importance(self, importance_df):
        """
        Plot the mutual information and feature importance.

        Parameters
        ----------
        importance_df : pd.DataFrame
            DataFrame containing mutual information and feature importance values.
        """
        importance_melted = importance_df.melt(
            id_vars="Feature",
            value_vars=["MutualInfoRank", "FeatureImportanceRank"],
            var_name="variable",
            value_name="value",
        )

        # Rename the variables for better readability
        importance_melted["variable"] = importance_melted["variable"].map(
            {"MutualInfoRank": "MI Scores", "FeatureImportanceRank": "GB Scores"}
        )

        # Plot using seaborn's catplot
        plt.figure(figsize=(12, 8))
        sns.catplot(
            data=importance_melted,
            x="value",
            y="Feature",
            hue="variable",
            kind="bar",
            height=6,
            aspect=1.5,
        )

        plt.title("Mutual Information and Feature Importance")
        plt.xlabel("Importance Rank")
        plt.ylabel("Feature")
        plt.tight_layout()
        plt.show()

    def plot_mi_gb_matrix(self, importance_df):
        """
        Plots a matrix categorizing features into four blocks based on their Mutual Information (MI) and Gradient Boosting (GB) scores.

        Parameters
        ----------
        importance_df : pd.DataFrame
            DataFrame containing mutual information and feature importance values.
        """
        high_mi = importance_df["MutualInfoRank"] > 0.5
        high_gb = importance_df["FeatureImportanceRank"] > 0.5

        matrix_data = {
            "High MI & High GB": importance_df[high_mi & high_gb],
            "High MI & Low GB": importance_df[high_mi & ~high_gb],
            "Low MI & High GB": importance_df[~high_mi & high_gb],
            "Low MI & Low GB": importance_df[~high_mi & ~high_gb],
        }

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        for ax, (title, data) in zip(axes.flatten(), matrix_data.items()):
            sns.barplot(
                x="FeatureImportanceRank",
                y="Feature",
                data=data,
                ax=ax,
                palette="viridis",
            )
            ax.set_title(title, fontsize=16)
            ax.set_xlim(0, 1)
            ax.set_xlabel("Feature Importance Rank", fontsize=14)
            ax.set_ylabel("Feature", fontsize=14)
            ax.tick_params(axis="both", which="major", labelsize=16)

        plt.tight_layout()
        plt.show()

    def determine_optimal_clusters(self, df, k_range=(1, 10), model_params=None):
        """
        Determines the optimal number of clusters using the Elbow method.

        Parameters
        ----------
        df : pd.DataFrame or np.ndarray
            The dataset to be analyzed.
        k_range : tuple, optional
            The range of cluster numbers to try (default is (1, 10)).
        model_params : dict, optional
            Additional parameters to pass to the KMeans model (default is None).
        """
        if not isinstance(df, pd.DataFrame):
            if isinstance(df, np.ndarray):
                df = pd.DataFrame(df)
            else:
                raise ValueError(
                    "Input data must be a pandas DataFrame or numpy ndarray."
                )

        X = df.select_dtypes(include=["int64", "float64"])

        if model_params is None:
            model_params = {}

        model = KMeans(**model_params)
        visualizer = KElbowVisualizer(model, k=k_range)
        visualizer.fit(X)
        visualizer.show()
        return visualizer.elbow_value_

    def perform_full_eda(self):
        """
        Performs full EDA by calling all the methods.
        """
        self.dataset_info()
        self.dataset_custom_info()
        self.dataset_shape()
        self.descriptive_statistics()
        self.missing_values()
        self.sample_data()
