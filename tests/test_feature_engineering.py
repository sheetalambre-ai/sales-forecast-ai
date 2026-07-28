"""
Unit tests for the feature engineering transformers.
"""

import pandas as pd
import pytest

from features.datetime import DateTimeFeatureTransformer
from features.lag import LagFeatureTransformer
from features.rolling import RollingFeatureTransformer
from features.pipeline import FeatureEngineeringPipeline


@pytest.fixture
def sample_dataframe():
    """
    Create a small sample dataset with store, item, date, and sales.
    """
    dates = pd.date_range(start="2025-01-01", periods=60, freq="D")
    df = pd.DataFrame(
        {
            "date": dates,
            "store": [1] * 60,
            "item": [1] * 60,
            "sales": list(range(100, 160)),
        }
    )
    return df


def test_datetime_feature_transformer(sample_dataframe):
    transformer = DateTimeFeatureTransformer()
    df_transformed = transformer.transform(sample_dataframe)

    expected_cols = [
        "year",
        "quarter",
        "month",
        "week",
        "day",
        "day_of_week",
        "day_of_year",
        "is_weekend",
        "is_month_start",
        "is_month_end",
    ]
    for col in expected_cols:
        assert col in df_transformed.columns
        assert not df_transformed[col].isnull().any()


def test_lag_feature_transformer(sample_dataframe):
    transformer = LagFeatureTransformer(lags=[1, 7])
    df_transformed = transformer.transform(sample_dataframe)

    assert "lag_1" in df_transformed.columns
    assert "lag_7" in df_transformed.columns

    # First row lag_1 should be NaN
    assert pd.isnull(df_transformed.loc[0, "lag_1"])
    # Second row lag_1 should be first row sales (100)
    assert df_transformed.loc[1, "lag_1"] == 100


def test_rolling_feature_transformer(sample_dataframe):
    transformer = RollingFeatureTransformer(windows=[7])
    df_transformed = transformer.transform(sample_dataframe)

    assert "rolling_mean_7" in df_transformed.columns
    assert "rolling_std_7" in df_transformed.columns

    # The rolling feature is computed on shifted sales, so first 7 rows mean should be NaN
    assert pd.isnull(df_transformed.loc[0:6, "rolling_mean_7"]).all()
    # Row 7 should have mean of first 7 rows (100 to 106)
    assert not pd.isnull(df_transformed.loc[7, "rolling_mean_7"])


def test_feature_engineering_pipeline(sample_dataframe):
    pipeline = FeatureEngineeringPipeline()
    # Let's override transformers to have smaller lags/windows to not drop too many rows
    pipeline.transformers = [
        DateTimeFeatureTransformer(),
        LagFeatureTransformer(lags=[1, 2]),
        RollingFeatureTransformer(windows=[3])
    ]
    df_transformed = pipeline.transform(sample_dataframe)

    # All generated columns should exist
    assert "year" in df_transformed.columns
    assert "lag_1" in df_transformed.columns
    assert "rolling_mean_3" in df_transformed.columns

    # Transformed data should drop NaNs and not be empty
    assert not df_transformed.empty
    assert not df_transformed.isnull().any().any()