import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler


class Discretisation(BaseEstimator, TransformerMixin):
    """
    Discretise the specified column into the specified number of bins.

    Methods
    -------
    fit(target)
        Return self
    transform(x: pd.DataFrame) -> pd.DataFrame
        Discretise the specified column into the specified number of bins.

    """

    def __init__(self, target: str, bins: int, labels: list[str]):
        """ Initialize the Discretisation object."""
        self.target = target
        self.bins = bins
        self.labels = labels

    def fit(self, target):
        """
        Parameters
        ----------
        target

        Returns
        -------
        self
        """
        return self

    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        """
        Discretise the specified column into the specified number of bins.

        Parameters
        ----------
        x

        Returns
        -------

        """
        x[self.target] = pd.cut(x[self.target],
                                bins=self.bins,
                                labels=self.labels)
        return x


class OneHotEncodePd(BaseEstimator, TransformerMixin):
    """
    One-hot encode the specified column.

    Methods
    -------
    fit(target)
        Return self
    transform(x: pd.DataFrame) -> pd.DataFrame
    """

    def __init__(self, target: str, prefix: str, sep: str, required_columns=None):
        """
        Initialize the OneHotEncodePd object.

        Parameters
        ----------
        target : str
            The column to one-hot encode.
        prefix : str
            The prefix to use for the one-hot encoded columns.
        sep : str
            The separator to use for the one-hot encoded columns.
        required_columns : list
            A list of columns that should be present in the DataFrame after one-hot encoding.
        """

        if required_columns is None:
            required_columns = []
        self.target = target
        self.prefix = prefix
        self.sep = sep
        self.required_columns = required_columns

    def fit(self, target):
        """

        Parameters
        ----------
        target

        Returns
        -------

        """
        return self

    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        """
        One-hot encode the specified column.

        Parameters
        ----------
        x

        Returns
        -------
        pd.DataFrame
        """
        # Perform in-place one-hot encoding
        df_encoded = pd.get_dummies(x, columns=[self.target], prefix=self.prefix,
                                    prefix_sep=self.sep, dtype=float)

        # Replace the original 'Category' column with the one-hot encoded columns
        x[df_encoded.columns] = df_encoded

        # Drop the original 'Category' column
        x.drop(columns=[self.target], inplace=True)

        # Ensure all required columns are present, adding them with 0s if necessary
        for column in self.required_columns:
            if column not in x.columns:
                x[column] = 0.0

        return x


class NormalizeCols(BaseEstimator, TransformerMixin):
    """
    Normalize the specified column to the specified feature range. 34

    Methods
    -------
    fit(target)
        Return self
    transform(x: pd.DataFrame) -> pd.DataFrame
        Normalize the specified column to the specified feature range.
    """

    def __init__(self, target: str, feature_range: tuple):
        """
        Initialize the NormalizeCols object.

        Parameters
        ----------
        target
        feature_range
        """
        self.target = target
        self.feature_range = feature_range

    def fit(self, target):
        """

        Parameters
        ----------
        target

        Returns
        -------
        self
        """
        return self

    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        """
    Normalize the specified column to the specified feature range.

    Parameters
    ----------
    x : pd.DataFrame
        The data to be transformed.

    Returns
    -------
    pd.DataFrame
        The transformed data.
    """
        df = x.copy()  # don't modify original df
        min_max_scaler = MinMaxScaler(feature_range=(self.feature_range[0], self.feature_range[1]))
        df[self.target] = min_max_scaler.fit_transform(df[[self.target]])
        return df
