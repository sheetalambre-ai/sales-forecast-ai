"""
Experiment logging utilities.

Logs model training experiments, metrics, hyperparameters,
and execution times for reproducibility.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from config.paths import RESULTS_DIR


class ExperimentLogger:
    """
    Records experiment metadata and saves it to disk.
    """

    def __init__(
        self,
        filename: str = "experiment_log.csv",
    ):
        self.filepath = RESULTS_DIR / filename

        self.records = []

    def log(
        self,
        model_name: str,
        metrics: Dict[str, Any],
        parameters: Optional[Dict[str, Any]] = None,
        training_time: Optional[float] = None,
        inference_time: Optional[float] = None,
        notes: str = "",
    ) -> None:
        """
        Record one experiment.
        """

        record = {
            "Timestamp": datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            ),
            "Model": model_name,
            "Training Time (s)": training_time,
            "Inference Time (s)": inference_time,
            "Notes": notes,
        }

        record.update(metrics)

        if parameters:

            for key, value in parameters.items():

                record[f"param_{key}"] = value

        self.records.append(record)

    def to_dataframe(
        self,
    ) -> pd.DataFrame:
        """
        Return experiment history as a DataFrame.
        """

        return pd.DataFrame(self.records)

    def save(self) -> Path:
        """
        Save experiments to CSV.

        If an experiment log already exists,
        append new experiments.
        """

        df = self.to_dataframe()

        if self.filepath.exists():

            old = pd.read_csv(self.filepath)

            df = pd.concat(
                [old, df],
                ignore_index=True,
            )

        df.to_csv(
            self.filepath,
            index=False,
        )

        return self.filepath

    def load(self) -> pd.DataFrame:
        """
        Load existing experiment history.
        """

        if not self.filepath.exists():

            return pd.DataFrame()

        return pd.read_csv(
            self.filepath,
        )

    def latest(self) -> pd.Series:
        """
        Return the latest logged experiment.
        """

        history = self.load()

        if history.empty:
            raise ValueError(
                "No experiments have been logged."
            )

        return history.iloc[-1]

    def clear(self) -> None:
        """
        Remove in-memory records.

        Does not delete the saved CSV.
        """

        self.records.clear()