import logging

import mlflow
import pandas as pd
from model.model_dev import (
    HyperparameterTuner,
    LightGBMModel,
    LinearRegressionModel,
    RandomForestModel,
    XGBoostModel,
)
from sklearn.base import RegressorMixin
from zenml import step
from zenml.client import Client

experiment_tracker = Client().active_stack.experiment_tracker


@step(experiment_tracker=experiment_tracker.name)
def train_model(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    model_name: str = "lightgbm",
    fine_tuning: bool = False,
    feature_set: str = "baseline",
    split_by: str = "order_id",
) -> RegressorMixin:
    try:
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("fine_tuning", fine_tuning)
        mlflow.log_param("feature_set", feature_set)
        mlflow.log_param("split_by", split_by)
        mlflow.log_param("n_features", x_train.shape[1])
        mlflow.set_tag("feature_columns", ",".join(x_train.columns))

        if model_name == "lightgbm":
            mlflow.lightgbm.autolog()
            model = LightGBMModel()
        elif model_name == "randomforest":
            mlflow.sklearn.autolog()
            model = RandomForestModel()
        elif model_name == "xgboost":
            mlflow.xgboost.autolog()
            model = XGBoostModel()
        elif model_name == "linear_regression":
            mlflow.sklearn.autolog()
            model = LinearRegressionModel()
        else:
            raise ValueError(f"Model name not supported: {model_name}")

        if fine_tuning:
            tuner = HyperparameterTuner(model, x_train, y_train, x_test, y_test)
            best_params = tuner.optimize()
            return model.train(x_train, y_train, **best_params)
        return model.train(x_train, y_train)
    except Exception as e:
        logging.error(e)
        raise e