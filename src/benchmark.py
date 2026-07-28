"""
Benchmark pipeline.

Runs the complete benchmarking workflow:
- Load data
- Feature engineering
- Train/test split
- Train models
- Evaluate models
- Log experiments
- Select best model
- Save reports
"""

from pathlib import Path
from typing import Optional

import pandas as pd

from data_pipeline.loader import load_dataset
from features.pipeline import FeatureEngineeringPipeline

from models import (
    ModelTrainer,
    get_models,
    train_test_split_time_series,
)

from experiment import (
    ExperimentLogger,
    BestModelSelector,
    ExperimentReport,
)

from explainability import (
    FeatureImportanceAnalyzer,
)

from config.constants import PRIMARY_METRIC


class BenchmarkPipeline:
    """
    End-to-end benchmarking pipeline.
    """

    def __init__(
        self,
        data_path: str | Path,
        enabled_models: Optional[list[str]] = None,
        model_kwargs: Optional[dict] = None,
    ):
        self.data_path = Path(data_path)
        self.enabled_models = enabled_models
        self.model_kwargs = model_kwargs or {}

        self.logger = ExperimentLogger()

        self.selector = BestModelSelector()

    def run(
        self,
    ) -> pd.DataFrame:
        """
        Execute benchmark pipeline.
        """

        print("=" * 70)
        print("Loading Dataset")
        print("=" * 70)

        df = load_dataset(
            self.data_path,
        )

        print(f"Dataset Shape : {df.shape}")

        print("\nApplying Feature Engineering...")

        pipeline = FeatureEngineeringPipeline()

        df = pipeline.fit_transform(df)

        print("Done")

        print("\nSplitting Dataset...")

        (
            X_train,
            X_test,
            y_train,
            y_test,
        ) = train_test_split_time_series(df)

        print("Done")

        # Instantiate selected models with custom kwargs
        from models.registry import AVAILABLE_MODELS, DEFAULT_MODEL_LIST
        
        models_to_train = []
        enabled = self.enabled_models if self.enabled_models is not None else DEFAULT_MODEL_LIST
        
        for name in enabled:
            if name not in AVAILABLE_MODELS:
                raise ValueError(f"Unknown model '{name}'")
            model_class = AVAILABLE_MODELS[name]
            kwargs = self.model_kwargs.get(name, {})
            models_to_train.append(model_class(**kwargs))

        trainer = ModelTrainer()

        trainer.benchmark(
            models_to_train,
            X_train,
            y_train,
            X_test,
            y_test,
        )

        results = trainer.get_results()

        print("\nBenchmark Results")

        print(results)

        # -------------------------------------------------
        # Log experiments
        # -------------------------------------------------

        for _, row in results.iterrows():

            metrics = row.to_dict()

            model_name = metrics.pop("Model")

            self.logger.log(
                model_name=model_name,
                metrics=metrics,
            )

        self.logger.save()

        # -------------------------------------------------
        # Best Model
        # -------------------------------------------------

        best = self.selector.best(results)

        print("\nBest Model")

        print(best)

        self.selector.save(results)

        # -------------------------------------------------
        # Experiment Report
        # -------------------------------------------------

        report = ExperimentReport(results)

        report.to_markdown()

        report.to_html()

        report.print_summary()

        # -------------------------------------------------
        # Save models
        # -------------------------------------------------

        trainer.save_all_models()

        # -------------------------------------------------
        # Feature Importance
        # -------------------------------------------------

        best_name = best["Model"]

        best_model = trainer.models.get(best_name)

        if (
            best_model is not None
            and hasattr(
                best_model,
                "feature_importances",
            )
        ):

            print(
                "\nGenerating Feature Importance..."
            )

            analyzer = FeatureImportanceAnalyzer(
                model=best_model,
                feature_names=X_train.columns,
            )

            analyzer.save_csv()

            analyzer.plot()

        return results