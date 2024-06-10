import pandas as pd
import os


class DataLoader:
    """
    A class to load datasets from either local files or URLs.

    Attributes
    ----------
    request_type : str
        Type of request, either 'local' for local files or 'url' for online CSV or PKL files.
    path : str
        Path to the local file or URL of the CSV or PKL file.

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
            Path to the local file or URL of the CSV or PKL file.
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
            # Determine the file extension
            file_extension = os.path.splitext(self.path)[1]
            if file_extension == ".csv":
                # Load data from a local CSV file
                return pd.read_csv(self.path)
            elif file_extension == ".pkl":
                # Load data from a local PKL file
                return pd.read_pickle(self.path)
            else:
                raise ValueError(
                    "Unsupported file type. Please use a '.csv' or '.pkl' file."
                )
        elif self.request_type == "url":
            # Determine the file extension
            file_extension = os.path.splitext(self.path)[1]
            if file_extension == ".csv":
                # Load data from a URL CSV file
                return pd.read_csv(self.path)
            elif file_extension == ".pkl":
                # Load data from a URL PKL file
                return pd.read_pickle(self.path)
            else:
                raise ValueError(
                    "Unsupported file type. Please use a '.csv' or '.pkl' file."
                )
        else:
            # Raise an error if the request type is invalid
            raise ValueError("Invalid request type. Please use 'local' or 'url'.")


# # Example usage
# if __name__ == "__main__":
#     # Load a local CSV file
#     csv_loader = DataLoader(request_type="local", path="path_to_your_dataset.csv")
#     df_csv = csv_loader.load_data()
#     print(df_csv.head())

#     # Load a local PKL file
#     pkl_loader = DataLoader(request_type="local", path="path_to_your_dataset.pkl")
#     df_pkl = pkl_loader.load_data()
#     print(df_pkl.head())
