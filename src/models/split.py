"""
Time series data splitting utilities.

This module provides utilities for splitting time series datasets into
training and testing sets while preserving chronological order.
"""

from typing import Tuple

import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from config.constants import DATE_COLUMN, TARGET_COLUMN
from config.settings import TEST_SIZE, TIME_SERIES_SPLITS


def validate_dataframe(df: pd.DataFrame) -> None:
    """
    Validate the input DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset.

    Raises
    ------
    ValueError
        If required columns are missing or the dataset is empty.
    """

    if df.empty:
        raise ValueError("Input DataFrame is empty.")

    if TARGET_COLUMN not in df.columns:
        raise ValueError(
            f"Target column '{TARGET_COLUMN}' not found."
        )

    if DATE_COLUMN not in df.columns:
        raise ValueError(
            f"Date column '{DATE_COLUMN}' not found."
        )


def prepare_features(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Separate features and target.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    X : pd.DataFrame
        Feature matrix.

    y : pd.Series
        Target vector.
    """

    drop_columns = [TARGET_COLUMN]

    if DATE_COLUMN in df.columns:
        drop_columns.append(DATE_COLUMN)

    X = df.drop(columns=drop_columns)

    y = df[TARGET_COLUMN]

    return X, y


def train_test_split_time_series(
    df: pd.DataFrame,
    test_size: float = TEST_SIZE,
) -> Tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.Series,
    pd.Series,
]:
    """
    Perform a chronological train/test split.

    Parameters
    ----------
    df : pd.DataFrame

    test_size : float

    Returns
    -------
    X_train
    X_test
    y_train
    y_test
    """

    validate_dataframe(df)

    df = df.sort_values(DATE_COLUMN).reset_index(drop=True)

    split_index = int(len(df) * (1 - test_size))

    train_df = df.iloc[:split_index]

    test_df = df.iloc[split_index:]

    X_train, y_train = prepare_features(train_df)

    X_test, y_test = prepare_features(test_df)

    return (
        X_train,
        X_test,
        y_train,
        y_test,
    )


def get_time_series_cv(
    n_splits: int = TIME_SERIES_SPLITS,
) -> TimeSeriesSplit:
    """
    Create a TimeSeriesSplit object.

    Parameters
    ----------
    n_splits : int

    Returns
    -------
    TimeSeriesSplit
    """

    return TimeSeriesSplit(
        n_splits=n_splits,
    )