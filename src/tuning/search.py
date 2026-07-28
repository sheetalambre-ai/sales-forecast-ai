"""
Hyperparameter search utilities.

Supports:
- Grid Search
- Randomized Search
- TimeSeriesSplit
- Any sklearn-compatible estimator
"""

from typing import Dict, Optional, Tuple

from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
)

from config.settings import (
    CV_SCORING,
    N_ITER_SEARCH,
)

from tuning.cv import get_time_series_cv


class HyperparameterSearch:
    """
    Generic hyperparameter tuning engine.
    """

    def __init__(
        self,
        estimator,
        param_grid: Dict,
        method: str = "random",
        scoring: str = CV_SCORING,
        n_iter: int = N_ITER_SEARCH,
        cv=None,
        random_state: int = 42,
        n_jobs: int = -1,
    ):
        """
        Parameters
        ----------
        estimator
            Model to optimize.

        param_grid
            Parameter search space.

        method
            "grid" or "random".

        scoring
            Scoring metric.

        n_iter
            Number of iterations for Random Search.

        cv
            Cross-validator.

        random_state
            Seed.

        n_jobs
            Number of parallel workers.
        """

        self.estimator = estimator
        self.param_grid = param_grid
        self.method = method.lower()
        self.scoring = scoring
        self.n_iter = n_iter
        self.cv = cv or get_time_series_cv()
        self.random_state = random_state
        self.n_jobs = n_jobs

        self.search = None

    def fit(
        self,
        X,
        y,
    ):
        """
        Perform hyperparameter search.
        """

        if self.method == "grid":

            self.search = GridSearchCV(
                estimator=self.estimator,
                param_grid=self.param_grid,
                scoring=self.scoring,
                cv=self.cv,
                n_jobs=self.n_jobs,
                refit=True,
            )

        elif self.method == "random":

            self.search = RandomizedSearchCV(
                estimator=self.estimator,
                param_distributions=self.param_grid,
                n_iter=self.n_iter,
                scoring=self.scoring,
                cv=self.cv,
                random_state=self.random_state,
                n_jobs=self.n_jobs,
                refit=True,
            )

        else:
            raise ValueError(
                "method must be either "
                "'grid' or 'random'."
            )

        self.search.fit(X, y)

        return self

    @property
    def best_estimator(self):
        """
        Return best fitted estimator.
        """

        if self.search is None:
            raise RuntimeError(
                "Search has not been fitted."
            )

        return self.search.best_estimator_

    @property
    def best_params(self):
        """
        Return best parameters.
        """

        if self.search is None:
            raise RuntimeError(
                "Search has not been fitted."
            )

        return self.search.best_params_

    @property
    def best_score(self):
        """
        Return best CV score.
        """

        if self.search is None:
            raise RuntimeError(
                "Search has not been fitted."
            )

        return self.search.best_score_

    def cv_results(self):
        """
        Return search results as a DataFrame.
        """

        if self.search is None:
            raise RuntimeError(
                "Search has not been fitted."
            )

        import pandas as pd

        return pd.DataFrame(
            self.search.cv_results_
        )