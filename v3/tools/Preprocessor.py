class Preprocessor:
    def __init__(self, preprocessor):
        """
        Initializes the preprocessor with the provided ColumnTransformer.

        Parameters
        ----------
        preprocessor : ColumnTransformer
            ColumnTransformer that includes transformers for numerical and categorical data.
        """
        self.preprocessor = preprocessor

    def fit_transform(self, X_train, X_val=None):
        """
        Applies preprocessing to the data.

        Parameters
        ----------
        X_train : pd.DataFrame
            Training data.
        X_val : pd.DataFrame, optional
            Validation data.

        Returns
        -------
        tuple: Transformed training and validation data.
        """
        X_train_transformed = self.preprocessor.fit_transform(X_train)
        if X_val is not None:
            X_val_transformed = self.preprocessor.transform(X_val)
            return X_train_transformed, X_val_transformed
        return X_train_transformed
