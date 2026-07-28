"""
Naive baseline forecasting model.
"""

import pandas as pd

from models.base import BaseForecastModel
from models.utils import save_model, load_model


class NaiveForecastModel(BaseForecastModel):
    """
    Predicts the last observed value from the training set.
    """

    def __init__(self):
        self.last_value = None

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
    ):
        self.last_value = y_train.iloc[-1]

    def predict(
        self,
        X_test: pd.DataFrame,
    ):

        return [self.last_value] * len(X_test)

    def save(self, filepath):
        save_model(self, filepath)

    def load(self, filepath):
        loaded = load_model(filepath)
        self.last_value = loaded.last_value