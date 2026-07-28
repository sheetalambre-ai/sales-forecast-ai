"""
Experiment report generation.

Creates Markdown and HTML reports summarizing
training results, best model, metrics, and timing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from config.constants import PRIMARY_METRIC
from config.paths import RESULTS_DIR


class ExperimentReport:
    """
    Generate experiment reports.
    """

    def __init__(
        self,
        results: pd.DataFrame,
    ):
        if results.empty:
            raise ValueError(
                "Results DataFrame is empty."
            )

        self.results = results.copy()

    @property
    def best_model(self) -> pd.Series:
        """
        Return the best-performing model.
        """

        ascending = PRIMARY_METRIC in {
            "MAE",
            "RMSE",
            "MAPE",
            "SMAPE",
            "MedianAE",
        }

        ranked = self.results.sort_values(
            by=PRIMARY_METRIC,
            ascending=ascending,
        )

        return ranked.iloc[0]

    def _df_to_markdown(self, df: pd.DataFrame) -> str:
        """
        Convert DataFrame to Markdown table, with fallback if tabulate is missing.
        """
        try:
            return df.to_markdown(index=False)
        except ImportError:
            headers = list(df.columns)
            header_row = "| " + " | ".join(map(str, headers)) + " |"
            sep_row = "| " + " | ".join(["---"] * len(headers)) + " |"
            body_rows = []
            for _, row in df.iterrows():
                formatted_vals = []
                for val in row:
                    if isinstance(val, float):
                        formatted_vals.append(f"{val:.4f}")
                    elif pd.isnull(val):
                        formatted_vals.append("")
                    else:
                        formatted_vals.append(str(val))
                body_rows.append("| " + " | ".join(formatted_vals) + " |")
            return "\n".join([header_row, sep_row] + body_rows)

    def to_markdown(
        self,
        filename: str = "experiment_report.md",
    ) -> Path:
        """
        Export report as Markdown.
        """

        output = RESULTS_DIR / filename

        best = self.best_model

        report = []

        report.append("# Experiment Report\n")

        report.append("## Best Model\n")

        report.append(
            f"- **Model:** {best['Model']}\n"
        )

        report.append(
            f"- **Primary Metric ({PRIMARY_METRIC}):** "
            f"{best[PRIMARY_METRIC]:.4f}\n"
        )

        report.append(
            "\n## Benchmark Results\n"
        )

        report.append(
            self._df_to_markdown(self.results)
        )

        output.write_text(
            "\n".join(report),
            encoding="utf-8",
        )

        return output


    def to_html(
        self,
        filename: str = "experiment_report.html",
    ) -> Path:
        """
        Export report as HTML.
        """

        output = RESULTS_DIR / filename

        best = self.best_model

        html = f"""
        <html>
        <head>
            <title>Experiment Report</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    margin: 40px;
                }}

                table {{
                    border-collapse: collapse;
                    width: 100%;
                }}

                th, td {{
                    border: 1px solid #ddd;
                    padding: 8px;
                }}

                th {{
                    background-color: #f2f2f2;
                }}

                h1 {{
                    color: #333;
                }}
            </style>
        </head>

        <body>

        <h1>Experiment Report</h1>

        <h2>Best Model</h2>

        <ul>
            <li><b>Model:</b> {best['Model']}</li>

            <li>
                <b>{PRIMARY_METRIC}:</b>
                {best[PRIMARY_METRIC]:.4f}
            </li>
        </ul>

        <h2>Benchmark Results</h2>

        {self.results.to_html(index=False)}

        </body>

        </html>
        """

        output.write_text(
            html,
            encoding="utf-8",
        )

        return output

    def print_summary(self) -> None:
        """
        Print experiment summary.
        """

        best = self.best_model

        print("\nExperiment Summary")
        print("=" * 60)

        print(
            f"Best Model : {best['Model']}"
        )

        print(
            f"{PRIMARY_METRIC:12}: "
            f"{best[PRIMARY_METRIC]:.4f}"
        )

        if "Training Time (s)" in best:
            print(
                f"Training Time : "
                f"{best['Training Time (s)']:.3f} s"
            )

        if "Inference Time (s)" in best:
            print(
                f"Inference Time : "
                f"{best['Inference Time (s)']:.3f} s"
            )

        print("\nRanking\n")

        print(
            self.results.to_string(
                index=False,
                float_format=lambda x: f"{x:.4f}",
            )
        )