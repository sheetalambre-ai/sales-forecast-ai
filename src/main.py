from pathlib import Path
from visualization.plots import generate_all_plots
from visualization.summary import print_summary
from visualization.report import save_summary_report
from data_pipeline.loader import load_dataset, get_dataset_info
from data_pipeline.validator import validate_dataset
from data_pipeline.preprocess import preprocess_dataset
from features.pipeline import FeatureEngineeringPipeline

RAW_DATA = Path("data/raw/train.csv")
PROCESSED_DATA = Path("data/processed/processed_sales.csv")


def main():

    print("=" * 50)
    print("SALES FORECAST AI")
    print("=" * 50)

    # Load
    df = load_dataset(RAW_DATA)

    # Summary
    get_dataset_info(df)

    # Validation
    validate_dataset(df)

    # Preprocess
    df = preprocess_dataset(df)

    # Save
    PROCESSED_DATA.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(PROCESSED_DATA, index=False)

    print("\nProcessed dataset saved successfully.")
    print(PROCESSED_DATA)

    print_summary(df)

    save_summary_report(df)

    generate_all_plots(df)

    pipeline = FeatureEngineeringPipeline()

    feature_df = pipeline.transform(df)

    pipeline.save(
        feature_df,
        "data/processed/sales_features.csv",
    )

if __name__ == "__main__":
    main()