import logging
from abc import ABC, abstractmethod
from typing import Union

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Optional, Union

from sklearn.model_selection import GroupShuffleSplit, train_test_split

class DataStrategy(ABC):
    """
    Abstract Class defining strategy for handling data
    """

    @abstractmethod
    def handle_data(self, data: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        pass


class DataPreprocessStrategy(DataStrategy):
    """
    Data preprocessing strategy which preprocesses the data.

    feature_set:
        "baseline"  - original numeric columns only
        "pre_order" - baseline + features known at purchase time
        "full"      - pre_order + actual delivery features (post-delivery only)
    """

    DATE_COLS = [
        "order_approved_at",
        "order_delivered_carrier_date",
        "order_delivered_customer_date",
        "order_estimated_delivery_date",
        "order_purchase_timestamp",
    ]

    def __init__(self, feature_set: str = "baseline") -> None:
        if feature_set not in ("baseline", "pre_order", "full"):
            raise ValueError(f"Unknown feature_set: {feature_set}")
        self.feature_set = feature_set

    @staticmethod
    def _days(later: pd.Series, earlier: pd.Series) -> pd.Series:
        return (later - earlier).dt.total_seconds() / 86400

    def _add_pre_order_features(self, data: pd.DataFrame) -> pd.DataFrame:
        purchase = data["order_purchase_timestamp"]
        data["estimated_delivery_days"] = self._days(
            data["order_estimated_delivery_date"], purchase
        )
        data["purchase_dayofweek"] = purchase.dt.dayofweek
        data["purchase_month"] = purchase.dt.month
        data["freight_ratio"] = data["freight_value"] / data["price"].replace(0, np.nan)
        data["volume_cm3"] = (
            data["product_length_cm"] * data["product_height_cm"] * data["product_width_cm"]
        )
        data["payment_per_installment"] = data["payment_value"] / data[
            "payment_installments"
        ].replace(0, np.nan)
        return data

    def _add_delivery_features(self, data: pd.DataFrame) -> pd.DataFrame:
        delivered = data["order_delivered_customer_date"]
        data["actual_delivery_days"] = self._days(
            delivered, data["order_purchase_timestamp"]
        )
        data["delivery_delay_days"] = self._days(
            delivered, data["order_estimated_delivery_date"]
        )
        data["is_late"] = (data["delivery_delay_days"] > 0).astype(int)
        return data

    def handle_data(self, data: pd.DataFrame) -> pd.DataFrame:
        try:
            data = data.copy()
            for col in self.DATE_COLS:
                data[col] = pd.to_datetime(data[col], errors="coerce")

            before = set(data.columns)
            if self.feature_set in ("pre_order", "full"):
                data = self._add_pre_order_features(data)
            if self.feature_set == "full":
                data = self._add_delivery_features(data)
            new_cols = [c for c in data.columns if c not in before]

            data = data.drop(self.DATE_COLS, axis=1)

            for col in [
                "product_weight_g",
                "product_length_cm",
                "product_height_cm",
                "product_width_cm",
            ]:
                data[col] = data[col].fillna(data[col].median())
            data["review_comment_message"] = data["review_comment_message"].fillna("No review")

            data = data.select_dtypes(include=[np.number])
            data = data.drop(["customer_zip_code_prefix", "order_item_id"], axis=1)

            # engineered columns: replace inf, then fill gaps with the median
            data = data.replace([np.inf, -np.inf], np.nan)
            for col in new_cols:
                if col in data.columns:
                    data[col] = data[col].fillna(data[col].median())

            return data
        except Exception as e:
            logging.error(e)
            raise e


class DataDivideStrategy(DataStrategy):
    """
    Divides the data into train and test sets. If `groups` is given, all rows
    sharing a group value (e.g. the same order_id) land on the same side.
    """

    def __init__(
        self,
        groups: Optional[pd.Series] = None,
        test_size: float = 0.2,
        random_state: int = 42,
    ) -> None:
        self.groups = groups
        self.test_size = test_size
        self.random_state = random_state

    def handle_data(self, data: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        try:
            X = data.drop("review_score", axis=1)
            y = data["review_score"]

            if self.groups is None:
                return train_test_split(
                    X, y, test_size=self.test_size, random_state=self.random_state
                )

            groups = self.groups.loc[data.index]
            splitter = GroupShuffleSplit(
                n_splits=1, test_size=self.test_size, random_state=self.random_state
            )
            train_idx, test_idx = next(splitter.split(X, y, groups=groups))
            return (
                X.iloc[train_idx],
                X.iloc[test_idx],
                y.iloc[train_idx],
                y.iloc[test_idx],
            )
        except Exception as e:
            logging.error(e)
            raise e


class DataCleaning:
    """
    Data cleaning class which preprocesses the data and divides it into train and test data.
    """

    def __init__(self, data: pd.DataFrame, strategy: DataStrategy) -> None:
        """Initializes the DataCleaning class with a specific strategy."""
        self.df = data
        self.strategy = strategy

    def handle_data(self) -> Union[pd.DataFrame, pd.Series]:
        """Handle data based on the provided strategy"""
        return self.strategy.handle_data(self.df)
