"""
Main entry point for the Sales Forecast AI project.

Usage:
    python src/main.py
"""

from pathlib import Path
import sys

from benchmark import BenchmarkPipeline

from config.paths import RAW_DATA_DIR
from utils.logger import get_logger

logger = get_logger(__name__)


def main() -> int:
    """
    Run the complete benchmarking pipeline.

    Returns
    -------
    int
        Exit status code.
    """

    try:

        dataset_path = RAW_DATA_DIR / "train.csv"

        logger.info("=" * 60)
        logger.info("Sales Forecast AI")
        logger.info("=" * 60)

        logger.info(
            "Dataset: %s",
            dataset_path,
        )

        pipeline = BenchmarkPipeline(
            data_path=dataset_path,
        )

        results = pipeline.run()

        logger.info("\nFinal Rankings\n")

        logger.info(
            "\n%s",
            results.to_string(index=False),
        )

        logger.info(
            "\nPipeline completed successfully."
        )

        return 0

    except FileNotFoundError as error:

        logger.exception(error)

        return 1

    except Exception as error:

        logger.exception(error)

        return 1


if __name__ == "__main__":

    sys.exit(main())