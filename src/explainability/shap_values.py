"""
SHAP explainability utilities.

Supports:
- SHAP summary plots
- SHAP bar plots
- SHAP dependence plots
- SHAP waterfall plots
- SHAP value export
"""

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd
import shap

from config.paths import FIGURES_DIR


class SHAPAnalyzer:
    """
    Generate SHAP explanations for tree-based models.
    """

    def __init__(
        self,
        model,
        X: pd.DataFrame,
    ):
        """
        Parameters
        ----------
        model
            Trained model wrapper exposing `.estimator`.

        X
            Feature matrix.
        """

        self.model = model.estimator
        self.X = X

        self.explainer = shap.TreeExplainer(
            self.model
        )

        self.shap_values = self.explainer.shap_values(
            self.X
        )

    def summary_plot(
        self,
        filename: str = "shap_summary.png",
        show: bool = False,
    ) -> Path:
        """
        Generate SHAP summary plot.
        """

        output = FIGURES_DIR / filename

        shap.summary_plot(
            self.shap_values,
            self.X,
            show=False,
        )

        plt.tight_layout()

        plt.savefig(
            output,
            dpi=300,
            bbox_inches="tight",
        )

        if show:
            plt.show()

        plt.close()

        return output

    def bar_plot(
        self,
        filename: str = "shap_bar.png",
        show: bool = False,
    ) -> Path:
        """
        Generate SHAP feature importance plot.
        """

        output = FIGURES_DIR / filename

        shap.summary_plot(
            self.shap_values,
            self.X,
            plot_type="bar",
            show=False,
        )

        plt.tight_layout()

        plt.savefig(
            output,
            dpi=300,
            bbox_inches="tight",
        )

        if show:
            plt.show()

        plt.close()

        return output

    def dependence_plot(
        self,
        feature_name: str,
        filename: Optional[str] = None,
        show: bool = False,
    ) -> Path:
        """
        Generate dependence plot for a feature.
        """

        if filename is None:
            filename = f"shap_dependence_{feature_name}.png"

        output = FIGURES_DIR / filename

        shap.dependence_plot(
            feature_name,
            self.shap_values,
            self.X,
            show=False,
        )

        plt.tight_layout()

        plt.savefig(
            output,
            dpi=300,
            bbox_inches="tight",
        )

        if show:
            plt.show()

        plt.close()

        return output

    def waterfall_plot(
        self,
        sample_index: int = 0,
        filename: str = "shap_waterfall.png",
        show: bool = False,
    ) -> Path:
        """
        Generate local explanation for one prediction.
        """

        output = FIGURES_DIR / filename

        explanation = shap.Explanation(
            values=self.shap_values[sample_index],
            base_values=self.explainer.expected_value,
            data=self.X.iloc[sample_index],
            feature_names=self.X.columns,
        )

        shap.plots.waterfall(
            explanation,
            show=False,
        )

        plt.tight_layout()

        plt.savefig(
            output,
            dpi=300,
            bbox_inches="tight",
        )

        if show:
            plt.show()

        plt.close()

        return output

    def export_values(
        self,
        filename: str = "shap_values.csv",
    ) -> Path:
        """
        Export SHAP values to CSV.
        """

        df = pd.DataFrame(
            self.shap_values,
            columns=self.X.columns,
        )

        output = FIGURES_DIR / filename

        df.to_csv(
            output,
            index=False,
        )

        return output