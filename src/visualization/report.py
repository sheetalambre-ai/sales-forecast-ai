"""
report.py

Save dataset summary to a text report.
"""

from pathlib import Path
import pandas as pd

from visualization.summary import (
    dataset_overview,
    numerical_summary,
    missing_value_summary,
    unique_value_summary,
    correlation_matrix,
)


def save_summary_report(df: pd.DataFrame):

    report_dir = Path("reports/summary")

    report_dir.mkdir(parents=True, exist_ok=True)

    report_path = report_dir / "summary.txt"

    with open(report_path, "w", encoding="utf-8") as file:

        file.write("=" * 60 + "\n")
        file.write("SALES FORECAST AI\n")
        file.write("=" * 60 + "\n\n")

        overview = dataset_overview(df)

        file.write("DATASET OVERVIEW\n")
        file.write("-" * 40 + "\n")

        for key, value in overview.items():
            file.write(f"{key}: {value}\n")

        file.write("\n")

        file.write("NUMERICAL SUMMARY\n")
        file.write("-" * 40 + "\n")
        file.write(numerical_summary(df).to_string())

        file.write("\n\n")

        file.write("MISSING VALUES\n")
        file.write("-" * 40 + "\n")
        file.write(missing_value_summary(df).to_string())

        file.write("\n\n")

        file.write("UNIQUE VALUES\n")
        file.write("-" * 40 + "\n")
        file.write(unique_value_summary(df).to_string())

        file.write("\n\n")

        file.write("CORRELATION MATRIX\n")
        file.write("-" * 40 + "\n")
        file.write(correlation_matrix(df).to_string())

    print("\nSummary report saved successfully.")

    print(report_path)