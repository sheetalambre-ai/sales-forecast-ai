"""
Unit tests for prediction and forecasting.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from models.linear_regression import LinearRegressionModel
from models.utils import save_model

from prediction.predict import Predictor
from prediction.forecast import Forecaster


@pytest.fixture
def trained_model(tmp_path: Path):
    """
    Train and save a simple model.
    """

    np.random.seed(42)

    X = pd.DataFrame(
        {
            "feature1": np.random.rand(50),
            "feature2": np.random.rand(50),
        }
    )

    y = (
        4 * X["feature1"]
        + 2 * X["feature2"]
        + np.random.normal(0, 0.05, 50)
    )

    model = LinearRegressionModel()

    model.train(X, y)

    model_path = tmp_path / "linear.pkl"

    save_model(
        model,
        model_path,
    )

    return (
        model_path,
        X,
        y,
    )


# ==========================================================
# Predictor
# ==========================================================

def test_predictor_load(
    trained_model,
):

    model_path, _, _ = trained_model

    predictor = Predictor(model_path)

    assert predictor.model is not None


def test_prediction_shape(
    trained_model,
):

    model_path, X, _ = trained_model

    predictor = Predictor(model_path)

    predictions = predictor.predict(X)

    assert len(predictions) == len(X)

    assert "Prediction" in predictions.columns


def test_prediction_with_actuals(
    trained_model,
):

    model_path, X, y = trained_model

    predictor = Predictor(model_path)

    comparison = predictor.predict_with_actuals(
        X,
        y,
    )

    assert "Actual" in comparison.columns

    assert "Prediction" in comparison.columns

    assert len(comparison) == len(y)


# ==========================================================
# CSV Export
# ==========================================================

def test_prediction_save(
    trained_model,
    tmp_path,
):

    model_path, X, _ = trained_model

    predictor = Predictor(model_path)

    output = predictor.predict_and_save(
        X,
        filename="predictions.csv",
    )

    assert output.exists()

    loaded = pd.read_csv(output)

    assert len(loaded) == len(X)


# ==========================================================
# Forecast
# ==========================================================

def test_forecast_generation(
    trained_model,
):

    model_path, X, _ = trained_model

    predictor = Predictor(model_path)

    forecaster = Forecaster(
        predictor,
    )

    forecast = forecaster.forecast(X)

    assert "Forecast" in forecast.columns

    assert len(forecast) == len(X)


def test_forecast_save(
    trained_model,
    tmp_path,
):

    model_path, X, _ = trained_model

    predictor = Predictor(model_path)

    forecaster = Forecaster(
        predictor,
    )

    forecast = forecaster.forecast(X)

    output = forecaster.save(
        forecast,
        filename="forecast.csv",
    )

    assert output.exists()


def test_forecast_plot(
    trained_model,
):

    model_path, X, _ = trained_model

    predictor = Predictor(model_path)

    forecaster = Forecaster(
        predictor,
    )

    forecast = forecaster.forecast(X)

    output = forecaster.plot(
        forecast,
        date_column=None,
    )

    assert output.exists()


# ==========================================================
# Invalid Input
# ==========================================================

def test_empty_dataframe(
    trained_model,
):

    model_path, _, _ = trained_model

    predictor = Predictor(model_path)

    empty = pd.DataFrame()

    with pytest.raises(ValueError):

        predictor.predict(empty)