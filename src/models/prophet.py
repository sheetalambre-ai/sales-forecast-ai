"""
Prophet forecasting model wrapper with fallback option.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from models.base import BaseForecastModel
from models.utils import save_model, load_model

try:
    from prophet import Prophet
    HAS_PROPHET = True
except ImportError:
    HAS_PROPHET = False


class ProphetModel(BaseForecastModel):
    """
    Prophet forecasting model wrapper. Falls back to an additive linear model
    if the prophet package is not installed.
    """

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.is_fallback = not HAS_PROPHET
        self.model = None

        if not self.is_fallback:
            self.model = Prophet(**self.kwargs)
        else:
            self.model = FallbackAdditiveModel()

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
    ) -> None:
        """
        Train the Prophet model.
        """
        # Reconstruct date column from year, month, day features
        if "year" in X_train.columns and "month" in X_train.columns and "day" in X_train.columns:
            ds = pd.to_datetime(pd.DataFrame({
                "year": X_train["year"],
                "month": X_train["month"],
                "day": X_train["day"]
            }))
        else:
            # Fallback to sequential daily range if calendar features are missing
            ds = pd.date_range(start="2020-01-01", periods=len(X_train), freq="D")

        ds = pd.Series(ds)

        if not self.is_fallback:
            train_df = pd.DataFrame({
                "ds": ds,
                "y": y_train
            })
            # Prophet outputs a lot of logs; run it quietly
            import logging
            logging.getLogger('prophet').setLevel(logging.WARNING)
            self.model.fit(train_df)
        else:
            self.model.fit(X_train, y_train, ds)

    def predict(
        self,
        X_test: pd.DataFrame,
    ):
        """
        Generate predictions.
        """
        if "year" in X_test.columns and "month" in X_test.columns and "day" in X_test.columns:
            ds = pd.to_datetime(pd.DataFrame({
                "year": X_test["year"],
                "month": X_test["month"],
                "day": X_test["day"]
            }))
        else:
            # Reconstruct sequence index or daily range
            ds = pd.date_range(start="2020-01-01", periods=len(X_test), freq="D")

        ds = pd.Series(ds)

        if not self.is_fallback:
            future_df = pd.DataFrame({"ds": ds})
            forecast = self.model.predict(future_df)
            return forecast["yhat"].values
        else:
            return self.model.predict(X_test, ds)

    def save(self, filepath: str) -> None:
        """
        Save the model.
        """
        save_model(
            {
                "model": self.model,
                "is_fallback": self.is_fallback,
                "kwargs": self.kwargs
            },
            filepath
        )

    def load(self, filepath: str) -> None:
        """
        Load the model.
        """
        data = load_model(filepath)
        self.model = data["model"]
        self.is_fallback = data["is_fallback"]
        self.kwargs = data["kwargs"]


class FallbackAdditiveModel:
    """
    Additive trend + seasonal model using LinearRegression.
    Fits:
    - Linear trend
    - Weekly seasonality (sine/cosine of day_of_week)
    - Yearly seasonality (sine/cosine of day_of_year)
    """

    def __init__(self):
        self.regressor = LinearRegression()
        self.start_date = None

    def _get_features(self, X: pd.DataFrame, ds: pd.Series) -> pd.DataFrame:
        features = pd.DataFrame(index=X.index)

        # 1. Trend feature (days since start_date)
        if self.start_date is None:
            self.start_date = ds.min()
        
        days_since_start = (ds - self.start_date).dt.total_seconds() / (24 * 3600)
        features["trend"] = days_since_start

        # 2. Weekly Seasonality
        if "day_of_week" in X.columns:
            dow = X["day_of_week"]
        else:
            dow = ds.dt.dayofweek
        
        features["weekly_sin"] = np.sin(2 * np.pi * dow / 7.0)
        features["weekly_cos"] = np.cos(2 * np.pi * dow / 7.0)

        # 3. Yearly Seasonality
        if "day_of_year" in X.columns:
            doy = X["day_of_year"]
        else:
            doy = ds.dt.dayofyear
        
        features["yearly_sin"] = np.sin(2 * np.pi * doy / 365.25)
        features["yearly_cos"] = np.cos(2 * np.pi * doy / 365.25)

        return features

    def fit(self, X: pd.DataFrame, y: pd.Series, ds: pd.Series):
        self.start_date = ds.min()
        features = self._get_features(X, ds)
        self.regressor.fit(features, y)

    def predict(self, X: pd.DataFrame, ds: pd.Series) -> np.ndarray:
        features = self._get_features(X, ds)
        return self.regressor.predict(features)
