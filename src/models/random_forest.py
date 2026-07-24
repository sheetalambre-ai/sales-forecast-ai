"""
Random Forest forecasting model.
"""

import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from models.base import BaseForecastModel
from models.utils import save_model, load_model


class RandomForestModel(BaseForecastModel):

    def __init__(
        self,
        n_estimators=200,
        random_state=42,
        n_jobs=-1,
    ):
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=n_jobs,
        )

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
    ):
        self.model.fit(X_train, y_train)

    def predict(
        self,
        X_test: pd.DataFrame,
    ):
        return self.model.predict(X_test)

    def save(self, filepath):
        save_model(self.model, filepath)

    def load(self, filepath):
        self.model = load_model(filepath)