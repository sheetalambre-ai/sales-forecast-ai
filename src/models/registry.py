"""
Model registry.

This module maintains the list of forecasting models available
for benchmarking and training.
"""

from typing import Dict, List

from models.baseline import NaiveForecastModel
from models.linear_regression import LinearRegressionModel
from models.random_forest import RandomForestModel
from models.xgboost_model import XGBoostModel
from models.prophet import ProphetModel
from models.lstm import LSTMModel

# =============================================================================
# Available Models
# =============================================================================

AVAILABLE_MODELS = {
    "baseline": NaiveForecastModel,
    "linear_regression": LinearRegressionModel,
    "random_forest": RandomForestModel,
    "xgboost": XGBoostModel,
    "prophet": ProphetModel,
    "lstm": LSTMModel,
}

# =============================================================================
# Default Benchmark Models
# =============================================================================

DEFAULT_MODEL_LIST = [
    "baseline",
    "linear_regression",
    "random_forest",
    "xgboost",
    "prophet",
    "lstm",
]


def get_available_models() -> Dict:
    """
    Return the dictionary of all registered models.

    Returns
    -------
    dict
        Mapping of model names to model classes.
    """

    return AVAILABLE_MODELS


def get_models(
    enabled_models: List[str] = None,
):
    """
    Instantiate forecasting models.

    Parameters
    ----------
    enabled_models : list[str], optional
        List of model names to create.
        If None, DEFAULT_MODEL_LIST is used.

    Returns
    -------
    list
        List of model objects.
    """

    if enabled_models is None:
        enabled_models = DEFAULT_MODEL_LIST

    models = []

    for model_name in enabled_models:

        if model_name not in AVAILABLE_MODELS:

            raise ValueError(
                f"Unknown model '{model_name}'."
            )

        model_class = AVAILABLE_MODELS[model_name]

        models.append(
            model_class()
        )

    return models


def list_models() -> List[str]:
    """
    Return the names of all registered models.
    """

    return list(AVAILABLE_MODELS.keys())