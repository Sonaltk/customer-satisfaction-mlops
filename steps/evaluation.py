import logging
from typing import Tuple

import mlflow
import pandas as pd
from model.evaluation import MSE, RMSE, R2Score
from sklearn.base import RegressorMixin
from typing_extensions import Annotated
from zenml import step
from zenml.client import Client

experiment_tracker = Client().active_stack.experiment_tracker


@step(experiment_tracker=experiment_tracker.name)
def evaluation(
    model: RegressorMixin, x_test: pd.DataFrame, y_test: pd.Series
) -> Tuple[
    Annotated[float, "mse"],
    Annotated[float, "r2_score"],
    Annotated[float, "rmse"],
]:
    try:
        prediction = model.predict(x_test)
        mse = float(MSE().calculate_score(y_test, prediction))
        r2 = float(R2Score().calculate_score(y_test, prediction))
        rmse = float(RMSE().calculate_score(y_test, prediction))
        mlflow.log_metric("mse", mse)
        mlflow.log_metric("r2_score", r2)
        mlflow.log_metric("rmse", rmse)
        return mse, r2, rmse
    except Exception as e:
        logging.error(e)
        raise e