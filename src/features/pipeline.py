"""
Feature engineering pipeline.
"""

from pathlib import Path
import pandas as pd

from features.datetime import DateTimeFeatureTransformer
from features.lag import LagFeatureTransformer
from features.rolling import RollingFeatureTransformer


class FeatureEngineeringPipeline:
    """
    Sequential feature engineering pipeline.
    """

    def __init__(self):

        self.transformers = [

            DateTimeFeatureTransformer(),

            LagFeatureTransformer(),

            RollingFeatureTransformer(),

        ]

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:

        transformed_df = df.copy()

        print("\nGenerating features...\n")

        for transformer in self.transformers:

            print(f"Running {transformer.__class__.__name__}")

            transformed_df = transformer.transform(transformed_df)

        transformed_df = transformed_df.dropna().reset_index(drop=True)

        print("\nFeature engineering completed.")

        return transformed_df

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fit and transform the dataset.
        """
        return self.transform(df)


    def save(
        self,
        df: pd.DataFrame,
        output_path: str,
    ):

        output = Path(output_path)

        output.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        df.to_csv(output, index=False)

        print(f"\nSaved engineered dataset:")
        print(output)