"""
Forecast generation utilities.

Provides functionality to:
- Generate future forecasts
- Create forecast DataFrames
- Save forecasts
- Plot forecasts
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd

from config.paths import FIGURES_DIR, RESULTS_DIR


class Forecaster:
    """
    Forecast generator using a trained predictor.
    """

    def __init__(
        self,
        predictor,
    ):
        """
        Parameters
        ----------
        predictor
            Predictor instance.
        """

        self.predictor = predictor

    def forecast(
        self,
        future_features: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Generate forecasts.

        Parameters
        ----------
        future_features
            Feature matrix representing future dates.

        Returns
        -------
        DataFrame
        """

        predictions = self.predictor.predict(
            future_features
        )

        forecast = future_features.copy()

        forecast["Forecast"] = predictions[
            "Prediction"
        ].values

        return forecast

    def save(
        self,
        forecast: pd.DataFrame,
        filename: str = "forecast.csv",
    ) -> Path:
        """
        Save forecast to CSV.
        """

        output = RESULTS_DIR / filename

        forecast.to_csv(
            output,
            index=False,
        )

        return output

    def plot(
        self,
        forecast: pd.DataFrame,
        date_column: Optional[str] = "date",
        value_column: str = "Forecast",
        filename: str = "forecast.png",
    ) -> Path:
        """
        Plot forecast values.
        """

        output = FIGURES_DIR / filename

        plt.figure(figsize=(12, 5))

        if date_column in forecast.columns:

            plt.plot(
                forecast[date_column],
                forecast[value_column],
                marker="o",
            )

            plt.xlabel(date_column)

        else:

            plt.plot(
                forecast[value_column],
                marker="o",
            )

            plt.xlabel("Time")

        plt.ylabel(value_column)

        plt.title("Forecast")

        plt.grid(True)

        plt.tight_layout()

        plt.savefig(
            output,
            dpi=300,
            bbox_inches="tight",
        )

        plt.close()

        return output