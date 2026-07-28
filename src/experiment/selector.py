"""
Best model selection utilities.

Provides functionality to:
- Rank experiment results
- Select the best-performing model
- Save best model metadata
"""

from pathlib import Path
from typing import Dict

import pandas as pd

from config.constants import PRIMARY_METRIC
from config.paths import RESULTS_DIR

# Metrics where lower values indicate better performance
LOWER_IS_BETTER = {
    "MAE",
    "RMSE",
    "MAPE",
    "SMAPE",
    "MedianAE",
}

# Metrics where higher values indicate better performance
HIGHER_IS_BETTER = {
    "R2",
    "ExplainedVariance",
}


class BestModelSelector:
    """
    Select the best-performing model from experiment results.
    """

    def __init__(
        self,
        metric: str = PRIMARY_METRIC,
    ):
        self.metric = metric

    def rank(
        self,
        results: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Rank experiment results.
        """

        if "Rank" in results.columns:
            results = results.drop(columns=["Rank"])

        if self.metric not in results.columns:

            raise ValueError(
                f"Metric '{self.metric}' not found."
            )

        if self.metric in LOWER_IS_BETTER:
            ascending = True

        elif self.metric in HIGHER_IS_BETTER:
            ascending = False

        else:
            raise ValueError(
                f"Unsupported metric '{self.metric}'."
            )

        ranked = (
            results.sort_values(
                by=self.metric,
                ascending=ascending,
            )
            .reset_index(drop=True)
        )

        ranked.insert(
            0,
            "Rank",
            range(1, len(ranked) + 1),
        )

        return ranked

    def best(
        self,
        results: pd.DataFrame,
    ) -> pd.Series:
        """
        Return the best model.
        """

        ranked = self.rank(results)

        return ranked.iloc[0]

    def save(
        self,
        results: pd.DataFrame,
        filename: str = "best_model.csv",
    ) -> Path:
        """
        Save the best model metadata.
        """

        best = self.best(results)

        output = RESULTS_DIR / filename

        best.to_frame().T.to_csv(
            output,
            index=False,
        )

        return output

    def summary(
        self,
        results: pd.DataFrame,
    ) -> None:
        """
        Print summary of the selected model.
        """

        best = self.best(results)

        print("\nBest Model")
        print("=" * 50)

        for column, value in best.items():
            print(f"{column:<25}: {value}")