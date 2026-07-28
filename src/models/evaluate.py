"""
Model evaluation utilities.

This module provides common regression metrics for evaluating
forecasting models.
"""

from typing import Dict

import numpy as np
from sklearn.metrics import (
    explained_variance_score,
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    r2_score,
)

from config.constants import SUPPORTED_METRICS


def mean_absolute_percentage_error(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    """
    Compute Mean Absolute Percentage Error (MAPE).

    Zero targets are ignored to avoid division by zero.
    """

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    mask = y_true != 0

    if mask.sum() == 0:
        return np.nan

    return (
        np.mean(
            np.abs(
                (y_true[mask] - y_pred[mask])
                / y_true[mask]
            )
        )
        * 100
    )


def symmetric_mean_absolute_percentage_error(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    """
    Compute Symmetric Mean Absolute Percentage Error (SMAPE).
    """

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    denominator = np.abs(y_true) + np.abs(y_pred)

    mask = denominator != 0

    if mask.sum() == 0:
        return np.nan

    return (
        np.mean(
            2
            * np.abs(y_pred[mask] - y_true[mask])
            / denominator[mask]
        )
        * 100
    )


def evaluate_model(
    y_true,
    y_pred,
) -> Dict[str, float]:
    """
    Evaluate regression predictions.

    Parameters
    ----------
    y_true : array-like

    y_pred : array-like

    Returns
    -------
    dict
        Dictionary containing evaluation metrics.
    """

    metrics = {

        "MAE": mean_absolute_error(
            y_true,
            y_pred,
        ),

        "RMSE": np.sqrt(
            mean_squared_error(
                y_true,
                y_pred,
            )
        ),

        "R2": r2_score(
            y_true,
            y_pred,
        ),

        "MAPE": mean_absolute_percentage_error(
            y_true,
            y_pred,
        ),

        "SMAPE": symmetric_mean_absolute_percentage_error(
            y_true,
            y_pred,
        ),

        "MedianAE": median_absolute_error(
            y_true,
            y_pred,
        ),

        "ExplainedVariance": explained_variance_score(
            y_true,
            y_pred,
        ),
    }

    return metrics


def print_metrics(
    metrics: Dict[str, float],
) -> None:
    """
    Print evaluation metrics in a readable format.
    """

    print("\nModel Performance")
    print("-" * 40)

    for metric in SUPPORTED_METRICS:

        if metric in metrics:

            print(
                f"{metric:<20}: {metrics[metric]:.4f}"
            )

    # Print additional metrics

    for metric in [

        "MedianAE",

        "ExplainedVariance",

    ]:

        print(
            f"{metric:<20}: {metrics[metric]:.4f}"
        )