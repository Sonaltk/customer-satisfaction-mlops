import logging
from typing import Tuple

import pandas as pd
from model.data_cleaning import (
    DataCleaning,
    DataDivideStrategy,
    DataPreprocessStrategy,
)
from typing_extensions import Annotated
from zenml import step


'''@step
def clean_data(
    data: pd.DataFrame,
    feature_set: str = "baseline",
) -> Tuple[
    Annotated[pd.DataFrame, "x_train"],
    Annotated[pd.DataFrame, "x_test"],
    Annotated[pd.Series, "y_train"],
    Annotated[pd.Series, "y_test"],
]:
    """Preprocess the data and split it into train and test sets."""
    try:
        preprocessed = DataCleaning(
            data, DataPreprocessStrategy(feature_set=feature_set)
        ).handle_data()
        x_train, x_test, y_train, y_test = DataCleaning(
            preprocessed, DataDivideStrategy()
        ).handle_data()
        return x_train, x_test, y_train, y_test
    except Exception as e:
        logging.error(e)
        raise e'''

@step
def clean_data(
    data: pd.DataFrame,
    feature_set: str = "baseline",
    split_by: str = "order_id",
) -> Tuple[
    Annotated[pd.DataFrame, "x_train"],
    Annotated[pd.DataFrame, "x_test"],
    Annotated[pd.Series, "y_train"],
    Annotated[pd.Series, "y_test"],
]:
    """Preprocess the data and split it into train and test sets."""
    try:
        groups = None if split_by == "none" else data[split_by]

        preprocessed = DataCleaning(
            data, DataPreprocessStrategy(feature_set=feature_set)
        ).handle_data()
        x_train, x_test, y_train, y_test = DataCleaning(
            preprocessed, DataDivideStrategy(groups=groups)
        ).handle_data()
        return x_train, x_test, y_train, y_test
    except Exception as e:
        logging.error(e)
        raise e