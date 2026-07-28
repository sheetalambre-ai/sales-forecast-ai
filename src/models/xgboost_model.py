"""
XGBoost forecasting model.

This module implements an XGBoost regressor that conforms to the
BaseForecastModel interface.
"""

from pathlib import Path
from typing import Optional

import pandas as pd
from xgboost import XGBRegressor

from config.model_params import XGBOOST_PARAMS
from models.base import BaseForecastModel
from models.utils import (
    load_model,
    save_model,
)


class XGBoostModel(BaseForecastModel):
    """
    XGBoost regression model.
    """

    def __init__(
        self,
        **kwargs,
    ):
        """
        Initialize the model.

        Parameters
        ----------
        kwargs : dict
            Parameters that override XGBOOST_PARAMS.
        """

        params = XGBOOST_PARAMS.copy()
        params.update(kwargs)

        self.model = XGBRegressor(
            **params,
        )

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_valid: Optional[pd.DataFrame] = None,
        y_valid: Optional[pd.Series] = None,
    ) -> None:
        """
        Train the XGBoost model.

        If a validation set is supplied, it is used as the evaluation set.
        """

        fit_kwargs = {}

        if X_valid is not None and y_valid is not None:
            fit_kwargs["eval_set"] = [
                (X_valid, y_valid)
            ]
            fit_kwargs["verbose"] = False

        self.model.fit(
            X_train,
            y_train,
            **fit_kwargs,
        )

    def predict(
        self,
        X_test: pd.DataFrame,
    ):
        """
        Generate predictions.
        """

        return self.model.predict(
            X_test,
        )

    def save(
        self,
        filepath: Path,
    ) -> None:
        """
        Save trained model.
        """

        save_model(
            self,
            filepath,
        )

    def load(
        self,
        filepath: Path,
    ) -> None:
        """
        Load trained model.
        """

        loaded = load_model(
            filepath,
        )
        self.model = loaded.model

    @property
    def feature_importances(self):
        """
        Return feature importances.
        """

        if not hasattr(
            self.model,
            "feature_importances_",
        ):
            raise AttributeError(
                "Model has not been trained yet."
            )

        return self.model.feature_importances_

    @property
    def estimator(self):
        """
        Return underlying XGBoost estimator.
        """

        return self.model