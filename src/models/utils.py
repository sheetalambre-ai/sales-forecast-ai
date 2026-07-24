"""
Model helper functions.
"""

from pathlib import Path
import joblib


def save_model(
    model,
    filepath: str,
):

    output = Path(filepath)

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        model,
        output,
    )


def load_model(
    filepath: str,
):

    return joblib.load(filepath)