"""
Time series train-test split utilities.
"""

import pandas as pd


def time_series_split(
    df: pd.DataFrame,
    target_column: str = "sales",
    test_size: float = 0.2,
):

    if target_column not in df.columns:
        raise ValueError(f"{target_column} not found.")

    split_index = int(len(df) * (1 - test_size))

    train = df.iloc[:split_index]

    test = df.iloc[split_index:]

    drop_columns = [target_column]

    if "date" in train.columns:
        drop_columns.append("date")

    X_train = train.drop(columns=drop_columns)

    y_train = train[target_column]

    X_test = test.drop(columns=drop_columns)

    y_test = test[target_column]

    return X_train, X_test, y_train, y_test