"""
Project path configuration.

This module defines all important project directories and file paths.
Using a centralized path configuration avoids hard-coded paths
throughout the project.
"""

from pathlib import Path

from config.constants import (
    DATA_DIR_NAME,
    RAW_DATA_DIR_NAME,
    PROCESSED_DATA_DIR_NAME,
    REPORTS_DIR_NAME,
    RESULTS_DIR_NAME,
    FIGURES_DIR_NAME,
    MODELS_DIR_NAME,
)

# =============================================================================
# Project Root
# =============================================================================

# Assumes this file is located at:
# project_root/src/config/paths.py

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SRC_DIR = PROJECT_ROOT / "src"

CONFIG_DIR = SRC_DIR / "config"

# =============================================================================
# Data Directories
# =============================================================================

DATA_DIR = PROJECT_ROOT / DATA_DIR_NAME

RAW_DATA_DIR = DATA_DIR / RAW_DATA_DIR_NAME

PROCESSED_DATA_DIR = DATA_DIR / PROCESSED_DATA_DIR_NAME

# =============================================================================
# Reports
# =============================================================================

REPORTS_DIR = PROJECT_ROOT / REPORTS_DIR_NAME

RESULTS_DIR = REPORTS_DIR / RESULTS_DIR_NAME

FIGURES_DIR = REPORTS_DIR / FIGURES_DIR_NAME

# =============================================================================
# Saved Models
# =============================================================================

SAVED_MODELS_DIR = PROJECT_ROOT / MODELS_DIR_NAME

# =============================================================================
# Logs
# =============================================================================

LOGS_DIR = PROJECT_ROOT / "logs"

# =============================================================================
# Tests
# =============================================================================

TESTS_DIR = PROJECT_ROOT / "tests"

# =============================================================================
# Application
# =============================================================================

APP_DIR = PROJECT_ROOT / "app"

# =============================================================================
# Ensure Required Directories Exist
# =============================================================================

DIRECTORIES = [
    DATA_DIR,
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    REPORTS_DIR,
    RESULTS_DIR,
    FIGURES_DIR,
    SAVED_MODELS_DIR,
    LOGS_DIR,
]

for directory in DIRECTORIES:
    directory.mkdir(
        parents=True,
        exist_ok=True,
    )