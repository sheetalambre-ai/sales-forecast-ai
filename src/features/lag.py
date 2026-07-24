"""
Lag feature engineering.
"""

import pandas as pd

from features.base import BaseFeatureTransformer


class LagFeatureTransformer(BaseFeatureTransformer):
    """
    Generate lag features grouped by entity.

    Default grouping:
        store + item
    """

    def __init__(
        self,
        target_column="sales",
        group_columns=None,
        lags=None,
    ):

        self.target_column = target_column

        self.group_columns = group_columns or [
            "store",
            "item",
        ]

        self.lags = lags or [1, 7, 14, 28]

    def transform(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:

        df = df.copy()

        for lag in self.lags:

            df[f"lag_{lag}"] = (

                df.groupby(self.group_columns)[
                    self.target_column
                ]

                .shift(lag)

            )

        return df