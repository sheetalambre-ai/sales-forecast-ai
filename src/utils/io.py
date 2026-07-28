"""
Input/output utilities.

Provides common file operations for:
- CSV
- JSON
- Pickle
- Directory creation
- Text files
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import pandas as pd


def ensure_directory(
    directory: Path | str,
) -> Path:
    """
    Create directory if it does not exist.
    """

    directory = Path(directory)

    directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    return directory


# ==========================================================
# CSV
# ==========================================================

def read_csv(
    filepath: Path | str,
    **kwargs,
) -> pd.DataFrame:
    """
    Read CSV file.
    """

    return pd.read_csv(
        filepath,
        **kwargs,
    )


def write_csv(
    dataframe: pd.DataFrame,
    filepath: Path | str,
    index: bool = False,
) -> Path:
    """
    Write DataFrame to CSV.
    """

    filepath = Path(filepath)

    ensure_directory(
        filepath.parent,
    )

    dataframe.to_csv(
        filepath,
        index=index,
    )

    return filepath


# ==========================================================
# JSON
# ==========================================================

def read_json(
    filepath: Path | str,
):
    """
    Read JSON file.
    """

    with open(
        filepath,
        "r",
        encoding="utf-8",
    ) as file:

        return json.load(file)


def write_json(
    data: Any,
    filepath: Path | str,
    indent: int = 4,
) -> Path:
    """
    Save object as JSON.
    """

    filepath = Path(filepath)

    ensure_directory(
        filepath.parent,
    )

    with open(
        filepath,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            data,
            file,
            indent=indent,
        )

    return filepath


# ==========================================================
# Pickle
# ==========================================================

def save_pickle(
    obj: Any,
    filepath: Path | str,
) -> Path:
    """
    Save Python object.
    """

    filepath = Path(filepath)

    ensure_directory(
        filepath.parent,
    )

    with open(
        filepath,
        "wb",
    ) as file:

        pickle.dump(
            obj,
            file,
        )

    return filepath


def load_pickle(
    filepath: Path | str,
):
    """
    Load Python object.
    """

    with open(
        filepath,
        "rb",
    ) as file:

        return pickle.load(file)


# ==========================================================
# Text
# ==========================================================

def write_text(
    text: str,
    filepath: Path | str,
) -> Path:
    """
    Write text file.
    """

    filepath = Path(filepath)

    ensure_directory(
        filepath.parent,
    )

    filepath.write_text(
        text,
        encoding="utf-8",
    )

    return filepath


def read_text(
    filepath: Path | str,
) -> str:
    """
    Read text file.
    """

    return Path(filepath).read_text(
        encoding="utf-8",
    )