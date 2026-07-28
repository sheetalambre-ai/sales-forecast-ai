"""
Unit tests for the data pipeline.
"""

from pathlib import Path

import pandas as pd
import pytest

from data_pipeline.loader import load_dataset
from data_pipeline.validator import validate_dataset
from data_pipeline.preprocess import preprocess_dataset


@pytest.fixture
def sample_dataframe():
    """
    Create a small sample dataset.
    """
    return pd.DataFrame(
        {
            "date": pd.date_range(
                start="2025-01-01",
                periods=5,
                freq="D",
            ),
            "sales": [100, 120, 110, 130, 125],
            "store": [1, 1, 1, 1, 1],
        }
    )


@pytest.fixture
def sample_csv(
    tmp_path: Path,
    sample_dataframe: pd.DataFrame,
):
    """
    Create a temporary CSV file.
    """
    file_path = tmp_path / "sample.csv"

    sample_dataframe.to_csv(
        file_path,
        index=False,
    )

    return file_path


# ==========================================================
# Loader Tests
# ==========================================================

def test_load_dataset(sample_csv):

    df = load_dataset(sample_csv)

    assert not df.empty

    assert len(df) == 5


def test_load_missing_file():

    with pytest.raises(FileNotFoundError):

        load_dataset("missing.csv")


# ==========================================================
# Validator Tests
# ==========================================================

def test_validate_dataset(sample_dataframe):

    assert validate_dataset(sample_dataframe)


def test_validate_empty_dataframe():

    empty = pd.DataFrame()

    with pytest.raises(ValueError):

        validate_dataset(empty)


# ==========================================================
# Preprocessing Tests
# ==========================================================

def test_preprocess_returns_dataframe(
    sample_dataframe,
):

    processed = preprocess_dataset(
        sample_dataframe
    )

    assert isinstance(
        processed,
        pd.DataFrame,
    )


def test_preprocess_preserves_rows(
    sample_dataframe,
):

    processed = preprocess_dataset(
        sample_dataframe
    )

    assert len(processed) == len(
        sample_dataframe
    )