import pandas as pd
import matplotlib.pyplot as plt


class EDA:
    """
    A class to perform Exploratory Data Analysis (EDA) on a dataset.

    Attributes
    ----------
    df : pd.DataFrame
        The dataset to be analyzed.

    Methods
    -------
    dataset_info():
        Prints information about the dataset.

    dataset_shape():
        Prints the shape of the dataset.

    descriptive_statistics():
        Displays descriptive statistics of the dataset.

    missing_values():
        Displays the count of missing values in the dataset.

    sample_data():
        Displays a sample of the dataset.

    plot_histogram(df, columns):
        Plots histograms for the specified columns.

    plot_scatter(df, x, y):
        Plots a scatter plot of the specified columns.

    perform_full_eda():
        Performs full EDA by calling all the methods.
    """

    def __init__(self, df):
        """
        Constructs all the necessary attributes for the EDA object.

        Parameters
        ----------
        df : pd.DataFrame
            The dataset to be analyzed.
        """
        self.df = df

    def dataset_info(self):
        """
        Prints information about the dataset.
        """
        print("Dataset Information:\n")
        print(self.df.info())

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

    def plot_histogram(self, df, columns):
        """
        Plots histograms for the specified columns.

        Parameters
        ----------
        df : pd.DataFrame
            The dataset containing the columns to plot.
        columns : list
            List of columns to plot histograms for.
        """
        for column in columns:
            plt.figure(figsize=(10, 6))
            df[column].hist(bins=30, edgecolor="black")
            plt.title(f"Histogram of {column}")
            plt.xlabel(column)
            plt.ylabel("Frequency")
            plt.show()

    def plot_scatter(self, df, x, y):
        """
        Plots a scatter plot of the specified columns.

        Parameters
        ----------
        df : pd.DataFrame
            The dataset containing the columns to plot.
        x : str
            The column to plot on the x-axis.
        y : str
            The column to plot on the y-axis.
        """
        plt.figure(figsize=(10, 6))
        plt.scatter(df[x], df[y], alpha=0.7)
        plt.title(f"Scatter Plot of {x} vs {y}")
        plt.xlabel(x)
        plt.ylabel(y)
        plt.show()

    def perform_full_eda(self):
        """
        Performs full EDA by calling all the methods.
        """
        self.dataset_info()
        self.dataset_shape()
        self.descriptive_statistics()
        self.missing_values()
        self.sample_data()
