import pandas as pd


class DataLoader:
    """
    A class to load datasets from either local files or URLs.

    Attributes
    ----------
    request_type : str
        Type of request, either 'local' for local files or 'url' for online CSV files.
    path : str
        Path to the local file or URL of the CSV file.

    Methods
    -------
    load_data():
        Loads the dataset based on the request type and path provided.
    """

    def __init__(self, request_type, path):
        """
        Constructs all the necessary attributes for the DataLoader object.

        Parameters
        ----------
        request_type : str
            Type of request, either 'local' or 'url'.
        path : str
            Path to the local file or URL of the CSV file.
        """
        self.request_type = request_type
        self.path = path

    def load_data(self):
        """
        Loads the dataset based on the request type and path provided.

        Returns
        -------
        pd.DataFrame
            Loaded dataset as a pandas DataFrame.

        Raises
        ------
        ValueError
            If the request type is neither 'local' nor 'url'.
        FileNotFoundError
            If the file is not found in the specified path.
        """
        if self.request_type == "local":
            # Load data from a local file
            return pd.read_csv(self.path)
        elif self.request_type == "url":
            # Load data from a URL
            return pd.read_csv(self.path)
        else:
            # Raise an error if the request type is invalid
            raise ValueError("Invalid request type. Please use 'local' or 'url'.")
