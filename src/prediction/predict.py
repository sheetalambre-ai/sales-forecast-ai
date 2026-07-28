"""
Prediction utilities.

Provides functionality to:
- Load trained models
- Validate input features
- Generate predictions
- Save prediction results
"""

from pathlib import Path
from typing import Union

import pandas as pd

from config.constants import TARGET_COLUMN
from config.paths import RESULTS_DIR, RAW_DATA_DIR
from models.utils import load_model


class Predictor:
    """
    Prediction engine for trained forecasting models.
    """

    def __init__(
        self,
        model_path: Union[str, Path],
    ):
        """
        Parameters
        ----------
        model_path
            Path to a saved model.
        """

        self.model_path = Path(model_path)

        # Dynamic model class lookup based on filename stem
        from models.baseline import NaiveForecastModel
        from models.linear_regression import LinearRegressionModel
        from models.random_forest import RandomForestModel
        from models.xgboost_model import XGBoostModel
        from models.prophet import ProphetModel
        from models.lstm import LSTMModel

        class_name_to_model = {
            "NaiveForecastModel": NaiveForecastModel,
            "LinearRegressionModel": LinearRegressionModel,
            "RandomForestModel": RandomForestModel,
            "XGBoostModel": XGBoostModel,
            "ProphetModel": ProphetModel,
            "LSTMModel": LSTMModel,
        }

        model_name = self.model_path.stem
        if model_name in class_name_to_model:
            model_class = class_name_to_model[model_name]
            self.model = model_class()
            self.model.load(self.model_path)
        else:
            self.model = load_model(self.model_path)

    def _prepare_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Automatically performs feature engineering if X contains raw columns
        (date, store, item) instead of pre-computed feature columns.
        """
        expected_columns = [
            'store', 'item', 'year', 'quarter', 'month', 'week', 'day', 
            'day_of_week', 'day_of_year', 'is_weekend', 'is_month_start', 
            'is_month_end', 'lag_1', 'lag_7', 'lag_14', 'lag_28', 
            'rolling_mean_7', 'rolling_std_7', 'rolling_mean_30', 'rolling_std_30'
        ]
        
        # Check if we already have the expected feature columns
        if all(col in X.columns for col in expected_columns):
            return X[expected_columns]
            
        # We need raw columns: date, store, item to engineer features.
        # If they are not present, this is a custom model/test dataset (e.g. feature1, feature2). Return X as-is.
        required_raw = ["date", "store", "item"]
        if not all(col in X.columns for col in required_raw):
            return X
            
        # Perform feature engineering using training history
        train_path = RAW_DATA_DIR / "train.csv"
        
        if not train_path.exists():
            raise FileNotFoundError(
                f"Training dataset not found at {train_path}. "
                "Historical sales are required to generate lag and rolling features."
            )
            
        # Load training dataset
        train_df = pd.read_csv(train_path)
        train_df["date"] = pd.to_datetime(train_df["date"])
        
        # Prepare input df
        X_df = X.copy()
        X_df["date"] = pd.to_datetime(X_df["date"])
        
        # Ensure sales column exists in X_df
        if "sales" not in X_df.columns:
            X_df["sales"] = 0.0
            
        # Track original row ordering
        X_df["_original_index"] = range(len(X_df))
        
        # Combine train_df and X_df
        cols = ["date", "store", "item", "sales"]
        combined_df = pd.concat([train_df[cols], X_df[cols]], ignore_index=True)
        
        # Sort chronologically per store/item so lag/rolling features are correct
        combined_df = combined_df.sort_values(["store", "item", "date"]).reset_index(drop=True)
        
        # Run feature engineering pipeline
        from features.pipeline import FeatureEngineeringPipeline
        pipeline = FeatureEngineeringPipeline()
        engineered_df = pipeline.fit_transform(combined_df)
        
        # Merge back to input DataFrame using date, store, item
        # First drop the sales column from engineered_df to avoid conflicts
        engineered_df = engineered_df.drop(columns=["sales"], errors="ignore")
        
        merged = pd.merge(
            X_df,
            engineered_df,
            on=["date", "store", "item"],
            how="left"
        )
        
        # Sort back to original index
        merged = merged.sort_values("_original_index").reset_index(drop=True)
        
        # Fill any remaining NaNs with 0
        merged = merged.fillna(0)
        
        # Return only the expected model features
        return merged[expected_columns]

    def predict(
        self,
        X: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Generate predictions.

        Parameters
        ----------
        X
            Feature dataframe.

        Returns
        -------
        pd.DataFrame
            Predictions.
        """

        if X.empty:
            raise ValueError(
                "Input data is empty."
            )

        X_prepared = self._prepare_features(X)
        predictions = self.model.predict(X_prepared)

        return pd.DataFrame(
            {
                "Prediction": predictions,
            }
        )

    def predict_and_save(
        self,
        X: pd.DataFrame,
        filename: str = "predictions.csv",
    ) -> Path:
        """
        Generate predictions and save to CSV.
        """

        predictions = self.predict(X)

        output = RESULTS_DIR / filename

        predictions.to_csv(
            output,
            index=False,
        )

        return output

    def predict_with_actuals(
        self,
        X: pd.DataFrame,
        y_true: pd.Series,
    ) -> pd.DataFrame:
        """
        Return actual and predicted values together.
        """

        predictions = self.predict(X)

        return pd.DataFrame(
            {
                "Actual": y_true.values,
                "Prediction": predictions[
                    "Prediction"
                ].values,
            }
        )