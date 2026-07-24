"""
Rolling feature engineering.
"""

import pandas as pd

from features.base import BaseFeatureTransformer


class RollingFeatureTransformer(BaseFeatureTransformer):

    def __init__(
        self,
        target_column="sales",
        group_columns=None,
        windows=None,
    ):

        self.target_column = target_column

        self.group_columns = group_columns or [
            "store",
            "item",
        ]

        self.windows = windows or [7, 30]

    def transform(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:

        df = df.copy()

        grouped = df.groupby(self.group_columns)[
            self.target_column
        ]

        for window in self.windows:

            shifted = grouped.shift(1)

            df[f"rolling_mean_{window}"] = (

                shifted

                .groupby(
                    [
                        df["store"],
                        df["item"],
                    ]
                )

                .rolling(window)

                .mean()

                .reset_index(
                    level=[0, 1],
                    drop=True,
                )

            )

            df[f"rolling_std_{window}"] = (

                shifted

                .groupby(
                    [
                        df["store"],
                        df["item"],
                    ]
                )

                .rolling(window)

                .std()

                .reset_index(
                    level=[0, 1],
                    drop=True,
                )

            )

        return df