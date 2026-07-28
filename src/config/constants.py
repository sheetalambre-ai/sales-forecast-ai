"""
Project-wide constants.

This module defines constant values that are used throughout the
sales forecasting project. These values are intended to remain
unchanged during normal execution.
"""

from pathlib import Path

# =============================================================================
# Project Information
# =============================================================================

PROJECT_NAME = "Sales Forecast AI"

PROJECT_VERSION = "1.0.0"

AUTHOR = "Sheetal Ambre"

LICENSE = "MIT"

# =============================================================================
# Randomness
# =============================================================================

DEFAULT_RANDOM_STATE = 42

# =============================================================================
# Dataset
# =============================================================================

TARGET_COLUMN = "sales"

DATE_COLUMN = "date"

STORE_COLUMN = "store"

ITEM_COLUMN = "item"

# =============================================================================
# Model Evaluation Metrics
# =============================================================================

PRIMARY_METRIC = "RMSE"

SUPPORTED_METRICS = (
    "MAE",
    "RMSE",
    "R2",
    "MAPE",
    "SMAPE",
)

# =============================================================================
# File Extensions
# =============================================================================

MODEL_EXTENSION = ".pkl"

CSV_EXTENSION = ".csv"

JSON_EXTENSION = ".json"

PNG_EXTENSION = ".png"

# =============================================================================
# Directory Names
# =============================================================================

DATA_DIR_NAME = "data"

RAW_DATA_DIR_NAME = "raw"

PROCESSED_DATA_DIR_NAME = "processed"

REPORTS_DIR_NAME = "reports"

RESULTS_DIR_NAME = "results"

FIGURES_DIR_NAME = "figures"

MODELS_DIR_NAME = "saved_models"

# =============================================================================
# Logging
# =============================================================================

LOG_FORMAT = (
    "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)

DEFAULT_LOG_LEVEL = "INFO"