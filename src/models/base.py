"""
Abstract base class for forecasting models.
"""

from abc import ABC, abstractmethod
import pandas as pd


class BaseForecastModel(ABC):
    """
    Base interface for all forecasting models.
    """

    @abstractmethod
    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
    ):
        pass

    @abstractmethod
    def predict(
        self,
        X_test: pd.DataFrame,
    ):
        pass

    @abstractmethod
    def save(
        self,
        filepath: str,
    ):
        pass

    @abstractmethod
    def load(
        self,
        filepath: str,
    ):
        pass