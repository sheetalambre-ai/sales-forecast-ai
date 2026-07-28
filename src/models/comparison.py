"""
Model comparison utilities.

This module provides utilities for comparing, ranking, and selecting
forecasting models based on evaluation metrics.
"""

from pathlib import Path
from typing import Optional

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


def rank_models(
    results: pd.DataFrame,
    metric: str = PRIMARY_METRIC,
) -> pd.DataFrame:
    """
    Rank models based on a given metric.

    Parameters
    ----------
    results : pd.DataFrame
        DataFrame containing evaluation metrics.

    metric : str
        Metric used for ranking.

    Returns
    -------
    pd.DataFrame
        Ranked DataFrame.
    """

    if "Rank" in results.columns:
        results = results.drop(columns=["Rank"])

    if metric not in results.columns:
        raise ValueError(
            f"'{metric}' not found in results."
        )


    if metric in LOWER_IS_BETTER:
        ascending = True

    elif metric in HIGHER_IS_BETTER:
        ascending = False

    else:
        raise ValueError(
            f"Unknown metric '{metric}'."
        )

    ranked = results.sort_values(
        by=metric,
        ascending=ascending,
    ).reset_index(drop=True)

    ranked.insert(
        0,
        "Rank",
        range(1, len(ranked) + 1),
    )

    return ranked


def get_best_model(
    results: pd.DataFrame,
    metric: str = PRIMARY_METRIC,
) -> pd.Series:
    """
    Return the best-performing model.
    """

    ranked = rank_models(
        results,
        metric,
    )

    return ranked.iloc[0]


def save_comparison(
    results: pd.DataFrame,
    filename: str = "comparison.csv",
) -> Path:
    """
    Save comparison results.

    Returns
    -------
    Path
        Output file path.
    """

    output_path = RESULTS_DIR / filename

    results.to_csv(
        output_path,
        index=False,
    )

    return output_path


def print_comparison(
    results: pd.DataFrame,
) -> None:
    """
    Print model comparison table.
    """

    print("\nModel Comparison")
    print("=" * 80)

    print(
        results.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )