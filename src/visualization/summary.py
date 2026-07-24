"""
summary.py

Generate summary statistics for exploratory data analysis.
"""

from pathlib import Path
import pandas as pd


def dataset_overview(df: pd.DataFrame) -> dict:
    """
    Return basic information about the dataset.
    """

    overview = {
        "Rows": df.shape[0],
        "Columns": df.shape[1],
        "Column Names": list(df.columns),
        "Missing Values": int(df.isnull().sum().sum()),
        "Duplicate Rows": int(df.duplicated().sum()),
    }

    return overview


def numerical_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return descriptive statistics for numerical columns.
    """

    return df.describe().round(2)


def missing_value_summary(df: pd.DataFrame) -> pd.Series:
    """
    Return missing values for every column.
    """

    return df.isnull().sum()


def unique_value_summary(df: pd.DataFrame) -> pd.Series:
    """
    Return number of unique values.
    """

    return df.nunique()


def correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute correlation matrix.
    """

    numeric_df = df.select_dtypes(include="number")

    return numeric_df.corr().round(2)


def print_summary(df: pd.DataFrame):

    print("\n" + "=" * 60)
    print("DATASET SUMMARY")
    print("=" * 60)

    overview = dataset_overview(df)

    for key, value in overview.items():
        print(f"{key:<20}: {value}")

    print("\nNumerical Summary")

    print(numerical_summary(df))

    print("\nMissing Values")

    print(missing_value_summary(df))

    print("\nUnique Values")

    print(unique_value_summary(df))

    print("\nCorrelation Matrix")

    print(correlation_matrix(df))