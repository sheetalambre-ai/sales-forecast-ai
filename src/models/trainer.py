"""
Training engine for forecasting models.

Responsibilities
----------------
1. Train models
2. Perform optional hyperparameter tuning
3. Evaluate predictions
4. Save trained models
5. Collect benchmark metrics
"""

from pathlib import Path
from time import perf_counter
from typing import List

import pandas as pd

from config.constants import PRIMARY_METRIC
from config.paths import SAVED_MODELS_DIR

from models.evaluate import evaluate_model
from models.comparison import rank_models


class ModelTrainer:
    """
    Generic training engine.
    """

    def __init__(self):

        self.results = []

        self.models = {}

    def train_model(
        self,
        model,
        X_train,
        y_train,
    ):
        """
        Train a single model.
        """

        start = perf_counter()

        model.train(
            X_train,
            y_train,
        )

        training_time = perf_counter() - start

        return training_time

    def evaluate_model(
        self,
        model,
        X_test,
        y_test,
    ):
        """
        Evaluate a trained model.
        """

        start = perf_counter()

        predictions = model.predict(
            X_test,
        )

        inference_time = perf_counter() - start

        metrics = evaluate_model(
            y_test,
            predictions,
        )

        metrics["Training Time (s)"] = None

        metrics["Inference Time (s)"] = inference_time

        return metrics

    def train_and_evaluate(
        self,
        model,
        X_train,
        y_train,
        X_test,
        y_test,
    ):
        """
        Complete training pipeline for one model.
        """

        print(f"\nTraining {model.__class__.__name__}")

        train_time = self.train_model(
            model,
            X_train,
            y_train,
        )

        metrics = self.evaluate_model(
            model,
            X_test,
            y_test,
        )

        metrics["Training Time (s)"] = train_time

        metrics["Model"] = model.__class__.__name__

        self.results.append(metrics)

        self.models[
            model.__class__.__name__
        ] = model

    def benchmark(
        self,
        model_list: List,
        X_train,
        y_train,
        X_test,
        y_test,
    ):
        """
        Train every model.
        """

        for model in model_list:

            self.train_and_evaluate(
                model,
                X_train,
                y_train,
                X_test,
                y_test,
            )

        return self.get_results()

    def get_results(
        self,
    ) -> pd.DataFrame:
        """
        Return benchmark table.
        """

        results = pd.DataFrame(
            self.results,
        )

        return rank_models(
            results,
            PRIMARY_METRIC,
        )

    def save_model(
        self,
        model_name,
    ):
        """
        Save one trained model.
        """

        model = self.models[model_name]

        output = (
            SAVED_MODELS_DIR
            / f"{model_name}.pkl"
        )

        model.save(
            output,
        )

    def save_all_models(
        self,
    ):
        """
        Save every trained model.
        """

        for name in self.models:

            self.save_model(name)