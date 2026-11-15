import logging

import pandas as pd
from model.data_cleaning import DataCleaning, DataPreprocessStrategy


def get_data_for_test(feature_set: str = "full", n: int = 100) -> str:
    """Sample n rows, preprocess them like the training data, return JSON."""
    try:
        df = pd.read_csv("./data/olist_customers_dataset.csv").sample(n=n, random_state=0)
        df = DataCleaning(df, DataPreprocessStrategy(feature_set=feature_set)).handle_data()
        df = df.drop(columns=["review_score"])
        return df.to_json(orient="split")
    except Exception as e:
        logging.error(e)
        raise e