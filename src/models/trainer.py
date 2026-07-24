"""
Generic model trainer.
"""

from pathlib import Path

import pandas as pd

from models.evaluate import evaluate_model


class ModelTrainer:

    def __init__(self):

        self.results = []

    def train_and_evaluate(

        self,

        model,

        X_train,

        y_train,

        X_test,

        y_test,

    ):

        print(f"\nTraining {model.__class__.__name__}")

        model.train(
            X_train,
            y_train,
        )

        predictions = model.predict(
            X_test,
        )

        metrics = evaluate_model(
            y_test,
            predictions,
        )

        metrics["Model"] = model.__class__.__name__

        self.results.append(metrics)

        return predictions

    def get_results(self):

        return pd.DataFrame(
            self.results
        )

    def save_results(
        self,
        filepath,
    ):

        output = Path(filepath)

        output.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.get_results().to_csv(
            output,
            index=False,
        )