"""
Unit tests for machine learning models.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from models.linear_regression import LinearRegressionModel
from models.random_forest import RandomForestModel
from models.baseline import NaiveForecastModel
from models.prophet import ProphetModel
from models.lstm import LSTMModel
from models.registry import get_models
from models.evaluate import evaluate_model
from models.utils import save_model, load_model


@pytest.fixture
def sample_dataset():
    """
    Small synthetic regression dataset.
    """

    np.random.seed(42)

    X = pd.DataFrame(
        {
            "feature1": np.random.rand(100),
            "feature2": np.random.rand(100),
            "feature3": np.random.rand(100),
        }
    )

    y = (
        5 * X["feature1"]
        + 3 * X["feature2"]
        + np.random.normal(0, 0.1, 100)
    )

    return X, y


# ==========================================================
# Linear Regression
# ==========================================================

def test_linear_regression_training(
    sample_dataset,
):

    X, y = sample_dataset

    model = LinearRegressionModel()

    model.train(X, y)

    predictions = model.predict(X)

    assert len(predictions) == len(y)


# ==========================================================
# Random Forest
# ==========================================================

def test_random_forest_training(
    sample_dataset,
):

    X, y = sample_dataset

    model = RandomForestModel()

    model.train(X, y)

    predictions = model.predict(X)

    assert len(predictions) == len(y)


# ==========================================================
# Baseline
# ==========================================================

def test_baseline_model(
    sample_dataset,
):

    X, y = sample_dataset

    model = NaiveForecastModel()

    model.train(X, y)

    predictions = model.predict(X)

    assert len(predictions) == len(y)


# ==========================================================
# Model Saving
# ==========================================================

def test_model_save_load(
    sample_dataset,
    tmp_path: Path,
):

    X, y = sample_dataset

    model = LinearRegressionModel()

    model.train(X, y)

    path = tmp_path / "model.pkl"

    save_model(
        model,
        path,
    )

    loaded = load_model(path)

    preds_original = model.predict(X)

    preds_loaded = loaded.predict(X)

    assert np.allclose(
        preds_original,
        preds_loaded,
    )


# ==========================================================
# Evaluator
# ==========================================================

def test_evaluator_metrics(
    sample_dataset,
):

    X, y = sample_dataset

    model = LinearRegressionModel()

    model.train(X, y)

    predictions = model.predict(X)

    metrics = evaluate_model(
        y,
        predictions,
    )

    assert "RMSE" in metrics

    assert "MAE" in metrics

    assert "R2" in metrics


# ==========================================================
# Registry
# ==========================================================

def test_model_registry():

    models = get_models()

    assert len(models) > 0

    names = [
        model.name
        for model in models
    ]

    assert len(names) == len(set(names))


# ==========================================================
# Prediction Shape
# ==========================================================

@pytest.mark.parametrize(
    "model_class",
    [
        LinearRegressionModel,
        RandomForestModel,
        NaiveForecastModel,
        ProphetModel,
        LSTMModel,
    ],
)

def test_prediction_shape(
    sample_dataset,
    model_class,
):

    X, y = sample_dataset

    model = model_class()

    model.train(X, y)

    predictions = model.predict(X)

    assert len(predictions) == len(X)


# ==========================================================
# Feature Importance
# ==========================================================

def test_random_forest_feature_importance(
    sample_dataset,
):

    X, y = sample_dataset

    model = RandomForestModel()

    model.train(X, y)

    importance = model.feature_importances

    assert len(importance) == X.shape[1]