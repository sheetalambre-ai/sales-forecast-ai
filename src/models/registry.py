from models.baseline import NaiveForecastModel
from models.linear_regression import LinearRegressionModel
from models.random_forest import RandomForestModel


def get_models():

    return [

        NaiveForecastModel(),

        LinearRegressionModel(),

        RandomForestModel(),

    ]