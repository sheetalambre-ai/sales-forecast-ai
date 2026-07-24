"""
Linear Regression forecasting model.
"""

import pandas as pd

from sklearn.linear_model import LinearRegression

from models.base import BaseForecastModel
from models.utils import save_model, load_model


class LinearRegressionModel(BaseForecastModel):

    def __init__(self):

        self.model = LinearRegression()

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
    ):

        self.model.fit(
            X_train,
            y_train,
        )

    def predict(
        self,
        X_test: pd.DataFrame,
    ):

        return self.model.predict(X_test)

    def save(
        self,
        filepath,
    ):

        save_model(
            self.model,
            filepath,
        )

    def load(
        self,
        filepath,
    ):

        self.model = load_model(filepath)