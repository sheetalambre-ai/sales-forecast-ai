"""
Datetime feature engineering.
"""

import pandas as pd

from features.base import BaseFeatureTransformer


class DateTimeFeatureTransformer(BaseFeatureTransformer):
    """
    Generate calendar-based features.
    """

    def __init__(self, date_column: str = "date"):
        self.date_column = date_column

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:

        if self.date_column not in df.columns:
            raise ValueError(
                f"{self.date_column} column not found."
            )

        df = df.copy()

        date = pd.to_datetime(df[self.date_column])

        # Calendar Features
        df["year"] = date.dt.year
        df["quarter"] = date.dt.quarter
        df["month"] = date.dt.month
        df["week"] = date.dt.isocalendar().week.astype(int)

        df["day"] = date.dt.day

        df["day_of_week"] = date.dt.dayofweek

        df["day_of_year"] = date.dt.dayofyear

        # Weekend

        df["is_weekend"] = (
            df["day_of_week"] >= 5
        ).astype(int)

        # Month

        df["is_month_start"] = (
            date.dt.is_month_start
        ).astype(int)

        df["is_month_end"] = (
            date.dt.is_month_end
        ).astype(int)

        return df