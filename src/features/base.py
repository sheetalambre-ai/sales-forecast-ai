"""
Base feature transformer.

Every feature engineering module should inherit from this class.
"""

from abc import ABC, abstractmethod
import pandas as pd


class BaseFeatureTransformer(ABC):
    """
    Base class for all feature transformers.
    """

    @abstractmethod
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply feature engineering transformation.

        Parameters
        ----------
        df : pd.DataFrame

        Returns
        -------
        pd.DataFrame
        """
        pass