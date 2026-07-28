"""
Feature importance utilities.

Provides functionality to:
- Extract feature importance from tree-based models
- Rank features
- Save feature importance to CSV
- Generate bar plots
"""

from pathlib import Path
from typing import Union

import matplotlib.pyplot as plt
import pandas as pd

from config.paths import FIGURES_DIR, RESULTS_DIR


class FeatureImportanceAnalyzer:
    """
    Analyze and visualize feature importance for supported models.
    """

    def __init__(
        self,
        model,
        feature_names,
    ):
        """
        Parameters
        ----------
        model
            Trained model wrapper.

        feature_names
            List of feature names.
        """

        self.model = model
        self.feature_names = list(feature_names)

    def get_importance(self) -> pd.DataFrame:
        """
        Return ranked feature importances.

        Returns
        -------
        pd.DataFrame
        """

        if not hasattr(
            self.model,
            "feature_importances",
        ):
            raise AttributeError(
                "Model does not expose feature importances."
            )

        importance = self.model.feature_importances

        df = pd.DataFrame(
            {
                "Feature": self.feature_names,
                "Importance": importance,
            }
        )

        df = df.sort_values(
            by="Importance",
            ascending=False,
        ).reset_index(drop=True)

        return df

    def save_csv(
        self,
        filename: str = "feature_importance.csv",
    ) -> Path:
        """
        Save importance rankings to CSV.
        """

        df = self.get_importance()

        output = RESULTS_DIR / filename

        df.to_csv(
            output,
            index=False,
        )

        return output

    def plot(
        self,
        top_n: int = 20,
        figsize=(10, 6),
        filename: str = "feature_importance.png",
    ) -> Path:
        """
        Plot feature importance.

        Parameters
        ----------
        top_n
            Number of top features to display.
        """

        df = self.get_importance().head(top_n)

        plt.figure(figsize=figsize)

        plt.barh(
            df["Feature"],
            df["Importance"],
        )

        plt.gca().invert_yaxis()

        plt.xlabel("Importance")

        plt.ylabel("Feature")

        plt.title("Feature Importance")

        plt.tight_layout()

        output = FIGURES_DIR / filename

        plt.savefig(
            output,
            dpi=300,
            bbox_inches="tight",
        )

        plt.close()

        return output

    def summary(self) -> None:
        """
        Print feature importance table.
        """

        print("\nFeature Importance")
        print("-" * 60)

        print(
            self.get_importance().to_string(
                index=False,
                float_format=lambda x: f"{x:.4f}",
            )
        )