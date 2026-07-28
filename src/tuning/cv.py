"""
Cross-validation utilities for time series forecasting.

This module provides reusable cross-validation strategies for
forecasting models while preserving temporal ordering.
"""

from typing import Iterator, Tuple

import numpy as np
from sklearn.model_selection import TimeSeriesSplit

from config.settings import TIME_SERIES_SPLITS


class TimeSeriesCV:
    """
    Wrapper around sklearn's TimeSeriesSplit.
    """

    def __init__(
        self,
        n_splits: int = TIME_SERIES_SPLITS,
    ):
        """
        Parameters
        ----------
        n_splits : int
            Number of cross-validation folds.
        """

        self.n_splits = n_splits
        self.cv = TimeSeriesSplit(
            n_splits=n_splits,
        )

    def split(
        self,
        X,
    ) -> Iterator[
        Tuple[np.ndarray, np.ndarray]
    ]:
        """
        Generate train/test indices.

        Parameters
        ----------
        X : array-like

        Yields
        ------
        train_indices
        test_indices
        """

        yield from self.cv.split(X)

    def get_n_splits(self) -> int:
        """
        Return number of CV folds.
        """

        return self.cv.get_n_splits()


def get_time_series_cv(
    n_splits: int = TIME_SERIES_SPLITS,
) -> TimeSeriesSplit:
    """
    Return a configured TimeSeriesSplit object.

    Parameters
    ----------
    n_splits : int

    Returns
    -------
    TimeSeriesSplit
    """

    return TimeSeriesSplit(
        n_splits=n_splits,
    )