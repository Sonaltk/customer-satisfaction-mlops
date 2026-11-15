import os

import numpy as np
import pandas as pd
import pytest
from model.data_cleaning import (
    DataCleaning,
    DataDivideStrategy,
    DataPreprocessStrategy,
)

FMT = "%Y-%m-%d %H:%M:%S"


def make_raw(n_orders: int = 200, seed: int = 0) -> pd.DataFrame:
    """Synthetic data shaped like the merged Olist table: some orders have
    several item rows that share the same review score."""
    rng = np.random.RandomState(seed)
    rows = []
    for i in range(n_orders):
        purchase = pd.Timestamp("2018-01-01") + pd.Timedelta(days=int(rng.randint(0, 200)))
        estimated = purchase + pd.Timedelta(days=int(rng.randint(8, 30)))
        delivered = estimated + pd.Timedelta(days=int(rng.randint(-6, 10)))
        score = int(rng.choice([1, 2, 3, 4, 5], p=[0.1, 0.05, 0.1, 0.2, 0.55]))
        for item in range(int(rng.choice([1, 1, 1, 2, 3]))):
            rows.append(
                {
                    "order_id": f"order_{i}",
                    "order_purchase_timestamp": purchase.strftime(FMT),
                    "order_approved_at": purchase.strftime(FMT),
                    "order_delivered_carrier_date": purchase.strftime(FMT),
                    "order_delivered_customer_date": delivered.strftime(FMT),
                    "order_estimated_delivery_date": estimated.strftime(FMT),
                    "customer_zip_code_prefix": 1000,
                    "order_item_id": item + 1,
                    "review_comment_message": "ok",
                    "review_score": score,
                    "payment_sequential": 1,
                    "payment_installments": int(rng.randint(1, 6)),
                    "payment_value": float(rng.uniform(20, 300)),
                    "price": float(rng.uniform(10, 250)),
                    "freight_value": float(rng.uniform(5, 40)),
                    "product_name_lenght": float(rng.randint(10, 60)),
                    "product_description_lenght": float(rng.randint(50, 900)),
                    "product_photos_qty": float(rng.randint(1, 6)),
                    "product_weight_g": float(rng.randint(100, 5000)),
                    "product_length_cm": float(rng.randint(10, 60)),
                    "product_height_cm": float(rng.randint(5, 40)),
                    "product_width_cm": float(rng.randint(5, 40)),
                }
            )
    df = pd.DataFrame(rows)
    df.loc[0, "product_weight_g"] = np.nan  # missing value to be imputed
    df.loc[1, "payment_installments"] = 0   # would divide by zero
    return df


def preprocess(raw: pd.DataFrame, feature_set: str) -> pd.DataFrame:
    return DataCleaning(raw, DataPreprocessStrategy(feature_set=feature_set)).handle_data()


@pytest.mark.parametrize(
    "feature_set, n_features", [("baseline", 12), ("pre_order", 18), ("full", 21)]
)
def test_feature_counts(feature_set, n_features):
    df = preprocess(make_raw(), feature_set)
    assert df.drop(columns=["review_score"]).shape[1] == n_features


@pytest.mark.parametrize("feature_set", ["baseline", "pre_order", "full"])
def test_no_missing_or_infinite_values(feature_set):
    x = preprocess(make_raw(), feature_set).drop(columns=["review_score"])
    assert not x.isna().any().any()
    assert np.isfinite(x.to_numpy(dtype=float)).all()


def test_pre_order_features_exclude_actual_delivery():
    """Guard against leakage: pre-order features must not use delivery outcomes."""
    cols = set(preprocess(make_raw(), "pre_order").columns)
    assert not cols & {"actual_delivery_days", "delivery_delay_days", "is_late"}


def test_full_features_include_delivery_outcomes():
    cols = set(preprocess(make_raw(), "full").columns)
    assert {"actual_delivery_days", "delivery_delay_days", "is_late"} <= cols


def test_delivery_delay_values():
    raw = make_raw(n_orders=1).iloc[[0]].copy()
    raw["order_purchase_timestamp"] = "2018-03-01 00:00:00"
    raw["order_estimated_delivery_date"] = "2018-03-15 00:00:00"
    raw["order_delivered_customer_date"] = "2018-03-18 00:00:00"  # 3 days late
    row = preprocess(raw, "full").iloc[0]
    assert row["delivery_delay_days"] == pytest.approx(3.0)
    assert row["actual_delivery_days"] == pytest.approx(17.0)
    assert row["estimated_delivery_days"] == pytest.approx(14.0)
    assert row["is_late"] == 1


def test_unknown_feature_set_raises():
    with pytest.raises(ValueError):
        DataPreprocessStrategy(feature_set="nonsense")


def test_target_is_kept_and_in_range():
    df = preprocess(make_raw(), "full")
    assert df["review_score"].between(1, 5).all()


def split(raw, feature_set="full", grouped=True):
    data = preprocess(raw, feature_set)
    groups = raw["order_id"] if grouped else None
    return DataCleaning(data, DataDivideStrategy(groups=groups)).handle_data()


def orders_of(raw, part):
    return set(raw.loc[part.index, "order_id"])


def test_grouped_split_keeps_each_order_on_one_side():
    raw = make_raw()
    x_train, x_test, y_train, y_test = split(raw, grouped=True)
    assert orders_of(raw, x_train).isdisjoint(orders_of(raw, x_test))
    assert len(x_train) == len(y_train) and len(x_test) == len(y_test)


def test_random_split_leaks_orders():
    """Documents the original bug: a plain random split puts rows of the same
    order into both train and test."""
    raw = make_raw()
    x_train, x_test, _, _ = split(raw, grouped=False)
    assert len(orders_of(raw, x_train) & orders_of(raw, x_test)) > 0


def test_split_is_roughly_80_20():
    raw = make_raw(n_orders=500)
    x_train, x_test, _, _ = split(raw)
    assert 0.15 < len(x_test) / (len(x_train) + len(x_test)) < 0.25


DATA_PATH = "data/olist_customers_dataset.csv"


@pytest.mark.skipif(not os.path.exists(DATA_PATH), reason="dataset not available")
def test_real_data_grouped_split_has_no_overlap():
    raw = pd.read_csv(DATA_PATH)
    assert raw["order_id"].nunique() < len(raw)  # the duplicate-row situation
    x_train, x_test, _, _ = split(raw)
    assert orders_of(raw, x_train).isdisjoint(orders_of(raw, x_test))
    assert x_train.shape[1] == 21