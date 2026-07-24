"""
Model comparison utilities.
"""

import pandas as pd


def rank_models(results: pd.DataFrame):

    return results.sort_values(
        by="RMSE",
    )