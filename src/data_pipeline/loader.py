import pandas as pd
from pathlib import Path


def load_dataset(filepath: str) -> pd.DataFrame:
    """
    Load CSV dataset.

    Parameters
    ----------
    filepath : str
        Path to CSV file.

    Returns
    -------
    pd.DataFrame
    """
    path = Path(filepath)

    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {filepath}")

    df = pd.read_csv(path)

    print("Dataset loaded successfully.\n")

    return df


def get_dataset_info(df: pd.DataFrame) -> None:
    """Print basic dataset information."""

    print("=" * 50)
    print("DATASET SUMMARY")
    print("=" * 50)

    print(f"Rows    : {df.shape[0]}")
    print(f"Columns : {df.shape[1]}")

    print("\nColumn Names")
    print(df.columns.tolist())

    print("\nData Types")
    print(df.dtypes)