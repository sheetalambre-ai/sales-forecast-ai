import pandas as pd


def parse_dates(df: pd.DataFrame, date_column: str = "date") -> pd.DataFrame:
    """
    Convert date column to datetime.
    """

    if date_column in df.columns:
        df[date_column] = pd.to_datetime(df[date_column])

    return df


def sort_dataset(df: pd.DataFrame, date_column: str = "date") -> pd.DataFrame:
    """
    Sort dataframe by date.
    """

    if date_column in df.columns:
        df = df.sort_values(by=date_column)

    return df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove duplicate rows.
    """

    return df.drop_duplicates()


def preprocess_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Complete preprocessing pipeline.
    """

    df = parse_dates(df)

    df = remove_duplicates(df)

    df = sort_dataset(df)

    df.reset_index(drop=True, inplace=True)

    return df