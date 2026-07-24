"""
Evaluation metrics.
"""

import numpy as np

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


def evaluate_model(
    y_true,
    y_pred,
):

    mae = mean_absolute_error(
        y_true,
        y_pred,
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_true,
            y_pred,
        )
    )

    r2 = r2_score(
        y_true,
        y_pred,
    )

    mask = y_true != 0

    if mask.sum() > 0:

        mape = np.mean(
            np.abs(
                (
                    y_true[mask] - y_pred[mask]
                )
                / y_true[mask]
            )
        ) * 100

    else:

        mape = np.nan

    return {

        "MAE": mae,

        "RMSE": rmse,

        "R2": r2,

        "MAPE": mape,

    }