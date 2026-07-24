import pandas as pd

from models import (
    get_models,
    time_series_split,
)

from models.trainer import ModelTrainer
from models.comparison import rank_models


df = pd.read_csv(
    "../data/processed/sales_features.csv"
)

X_train, X_test, y_train, y_test = time_series_split(df)

trainer = ModelTrainer()

for model in get_models():

    trainer.train_and_evaluate(
        model,
        X_train,
        y_train,
        X_test,
        y_test,
    )

results = trainer.get_results()

results = rank_models(results)

print(results)

trainer.save_results(
    "../reports/results/model_results.csv"
)