import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler


class ColumnDropper(BaseEstimator, TransformerMixin):
    """
    Drop the specified columns from the DataFrame.

    Methods
    -------
    fit(target)
        Return self
    transform(x: pd.DataFrame) -> pd.DataFrame
        Drop the specified columns from the DataFrame.
    """
    def __init__(self, target: list):
        """ Initialize the ColumnDropper object."""
        self.target = target

    def fit(self, target):
        """ Return self."""
        return self

    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        """
        Drop the specified columns from the DataFrame.

        Parameters
        ----------
        x : pd.DataFrame
            The DataFrame to transform.
        """
        return x.drop(self.target, axis=1)


class DropIncompleteRow(BaseEstimator, TransformerMixin):
    """
    Drop rows with missing values in the specified columns.

    Methods
    -------
    fit(target)
        Return self
    transform(x: pd.DataFrame) -> pd.DataFrame
        Drop rows with missing values in the specified columns.
    """
    def __init__(self, target: list[str]):
        """ Initialize the DropIncompleteRow object."""
        self.target = target

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
        Drop rows with missing values in the specified columns.

        Parameters
        ----------
        x : pd.DataFrame

        Returns
        -------
        pd.DataFrame
        """
        return x.dropna(subset=self.target)


class FillNaNWithMean(BaseEstimator, TransformerMixin):
    """
    Fill missing values in the specified column with the mean of the column.

    Methods
    -------
    fit(target)
        Return self
    transform(x: pd.DataFrame) -> pd.DataFrame
        Fill missing values in the specified column with the mean of the column.
    """
    def __init__(self, target: str):
        """ Initialize the FillNaNWithMean object."""
        self.target = target

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
        Fill missing values in the specified column with the mean of the column.

        Parameters
        ----------
        x

        Returns
        -------
        pd.DataFrame
        """
        x[self.target].fillna(x[self.target].mean(), inplace=True)
        return x


class FillNaNWithMedian(BaseEstimator, TransformerMixin):
    """
    Fill missing values in the specified column with the median of the column.
    """
    def __init__(self, target: str):
        """ Initialize the FillNaNWithMedian object."""
        self.target = target

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
        Fill missing values in the specified column with the median of the column.

        Parameters
        ----------
        x

        Returns
        -------
        pd.DataFrame
        """
        x[self.target].fillna(x[self.target].median(), inplace=True)
        return x


class FillNaNWithValue(BaseEstimator, TransformerMixin):
    """
    Fill missing values in the specified column with the specified value.

    Methods
    -------
    fit(target)
        Return self
    transform(x: pd.DataFrame) -> pd.DataFrame
        Fill missing values in the specified column with the specified value.

    """
    def __init__(self, target: str, value: float):
        """ Initialize the FillNaNWithValue object."""
        self.target = target
        self.value = value

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
        Fill missing values in the specified column with the specified value.

        Parameters
        ----------
        x

        Returns
        -------
        pd.DataFrame
        """
        x[self.target].fillna(self.value, inplace=True)
        return x
