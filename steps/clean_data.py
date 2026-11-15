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
    Annotated[pd.Series, "groups_train"],
]:
    """Preprocess, split by group, and also return the training group ids."""
    try:
        groups = None if split_by == "none" else data[split_by]

        preprocessed = DataCleaning(
            data, DataPreprocessStrategy(feature_set=feature_set)
        ).handle_data()
        x_train, x_test, y_train, y_test = DataCleaning(
            preprocessed, DataDivideStrategy(groups=groups)
        ).handle_data()

        if groups is None:
            groups_train = pd.Series(x_train.index, index=x_train.index, name="group")
        else:
            groups_train = groups.loc[x_train.index]
        return x_train, x_test, y_train, y_test, groups_train
    except Exception as e:
        logging.error(e)
        raise e