import pandas as pd


def check_missing_values(df: pd.DataFrame):
    """Return missing value counts."""
    return df.isnull().sum()


def check_duplicates(df: pd.DataFrame):
    """Return duplicate row count."""
    return df.duplicated().sum()


def check_datatypes(df: pd.DataFrame):
    """Return column datatypes."""
    return df.dtypes


def validate_dataset(df: pd.DataFrame):
    """
    Validate dataset and print report.
    """

    print("\n" + "=" * 50)
    print("VALIDATION REPORT")
    print("=" * 50)

    print("\nMissing Values")

    missing = check_missing_values(df)

    if missing.sum() == 0:
        print("No missing values found.")
    else:
        print(missing)

    duplicates = check_duplicates(df)

    print(f"\nDuplicate Rows : {duplicates}")

    print("\nData Types")
    print(check_datatypes(df))