"""
Model configuration.

This module defines:

1. Default parameters for each model.
2. Hyperparameter search spaces for tuning.
"""

from config.settings import RANDOM_STATE

# =============================================================================
# Linear Regression
# =============================================================================

LINEAR_REGRESSION_PARAMS = {
    "fit_intercept": True,
}

# =============================================================================
# Random Forest
# =============================================================================

RANDOM_FOREST_PARAMS = {
    "n_estimators": 300,
    "max_depth": None,
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}

RANDOM_FOREST_SEARCH_SPACE = {
    "n_estimators": [100, 200, 300, 500],
    "max_depth": [10, 20, 30, None],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
}

# =============================================================================
# XGBoost
# =============================================================================

XGBOOST_PARAMS = {
    "objective": "reg:squarederror",
    "n_estimators": 300,
    "learning_rate": 0.05,
    "max_depth": 6,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}

XGBOOST_SEARCH_SPACE = {
    "n_estimators": [200, 400, 600],
    "learning_rate": [0.01, 0.05, 0.1],
    "max_depth": [4, 6, 8],
    "subsample": [0.8, 1.0],
    "colsample_bytree": [0.8, 1.0],
}

# =============================================================================
# LightGBM (Future)
# =============================================================================

LIGHTGBM_PARAMS = {
    "n_estimators": 300,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "random_state": RANDOM_STATE,
}

LIGHTGBM_SEARCH_SPACE = {
    "n_estimators": [200, 400, 600],
    "learning_rate": [0.01, 0.05, 0.1],
    "num_leaves": [31, 50, 100],
}

# =============================================================================
# CatBoost (Future)
# =============================================================================

CATBOOST_PARAMS = {
    "iterations": 300,
    "learning_rate": 0.05,
    "depth": 6,
    "random_state": RANDOM_STATE,
    "verbose": False,
}

CATBOOST_SEARCH_SPACE = {
    "iterations": [200, 400, 600],
    "learning_rate": [0.01, 0.05, 0.1],
    "depth": [4, 6, 8],
}