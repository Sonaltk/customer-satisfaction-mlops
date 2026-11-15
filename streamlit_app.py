import json
from datetime import date

import pandas as pd
import requests
import streamlit as st
from model.data_cleaning import DataPreprocessStrategy

DATA_PATH = "data/olist_customers_dataset.csv"
FEATURE_SET = "full"  # must match the feature set of the deployed model


@st.cache_data
def load_reference():
    """Preprocess the training data once to get the exact column order and
    dtypes the deployed model was trained on."""
    raw = pd.read_csv(DATA_PATH)
    ref = DataPreprocessStrategy(feature_set=FEATURE_SET).handle_data(raw)
    ref = ref.drop(columns=["review_score"])
    return list(ref.columns), ref.dtypes.to_dict()


def build_row(v: dict, columns: list, dtypes: dict) -> pd.DataFrame:
    """Turn raw order fields into the model's feature row, using the same
    preprocessing code as training."""
    purchase = pd.Timestamp(v["purchase_date"])
    raw = pd.DataFrame(
        [
            {
                "order_purchase_timestamp": purchase,
                "order_approved_at": purchase,
                "order_delivered_carrier_date": purchase,
                "order_delivered_customer_date": pd.Timestamp(v["delivered_date"]),
                "order_estimated_delivery_date": pd.Timestamp(v["estimated_date"]),
                # placeholders: dropped or ignored by the preprocessing
                "customer_zip_code_prefix": 0,
                "order_item_id": 1,
                "review_comment_message": "none",
                # real inputs
                "payment_sequential": v["payment_sequential"],
                "payment_installments": v["payment_installments"],
                "payment_value": v["payment_value"],
                "price": v["price"],
                "freight_value": v["freight_value"],
                "product_name_lenght": v["product_name_length"],
                "product_description_lenght": v["product_description_length"],
                "product_photos_qty": v["product_photos_qty"],
                "product_weight_g": v["product_weight_g"],
                "product_length_cm": v["product_length_cm"],
                "product_height_cm": v["product_height_cm"],
                "product_width_cm": v["product_width_cm"],
            }
        ]
    )
    row = DataPreprocessStrategy(feature_set=FEATURE_SET).handle_data(raw)
    return row[columns].astype(dtypes)


def predict(url: str, row: pd.DataFrame) -> float:
    payload = json.loads(row.to_json(orient="split"))
    body = {
        "dataframe_split": {"columns": payload["columns"], "data": payload["data"]}
    }
    resp = requests.post(url, json=body, timeout=10)
    resp.raise_for_status()
    out = resp.json()
    if isinstance(out, dict):
        out = out.get("predictions", out)
    return float(out[0])


st.set_page_config(page_title="Review score predictor", page_icon="⭐")
st.title("Customer review score predictor")
st.caption(
    "Predicts the review score (1-5) of an Olist order **after delivery**, "
    "using order, product and delivery information. Served by an MLflow model "
    "deployed through a ZenML pipeline."
)

endpoint = st.sidebar.text_input(
    "Model server endpoint", "http://127.0.0.1:8000/invocations"
)
st.sidebar.caption(
    "Start the server with `python run_deployment.py --config deploy`."
)

columns, dtypes = load_reference()

with st.form("order"):
    st.subheader("Delivery")
    d1, d2, d3 = st.columns(3)
    purchase_date = d1.date_input("Purchase date", date(2018, 3, 1))
    estimated_date = d2.date_input("Estimated delivery", date(2018, 3, 15))
    delivered_date = d3.date_input("Actual delivery", date(2018, 3, 10))

    st.subheader("Payment")
    p1, p2 = st.columns(2)
    price = p1.number_input("Price", min_value=0.01, value=100.0)
    freight_value = p2.number_input("Freight value", min_value=0.0, value=20.0)
    payment_value = p1.number_input("Total payment value", min_value=0.0, value=120.0)
    payment_installments = p2.number_input("Installments", min_value=1, value=1, step=1)
    payment_sequential = p1.number_input("Payment sequential", min_value=1, value=1, step=1)

    st.subheader("Product")
    q1, q2 = st.columns(2)
    name_len = q1.number_input("Name length (chars)", min_value=0, value=50, step=1)
    desc_len = q2.number_input("Description length (chars)", min_value=0, value=500, step=10)
    photos = q1.number_input("Number of photos", min_value=0, value=2, step=1)
    weight = q2.number_input("Weight (g)", min_value=0, value=1000, step=50)
    length_cm = q1.number_input("Length (cm)", min_value=1, value=30, step=1)
    height_cm = q2.number_input("Height (cm)", min_value=1, value=15, step=1)
    width_cm = q1.number_input("Width (cm)", min_value=1, value=20, step=1)

    submitted = st.form_submit_button("Predict")

if submitted:
    if delivered_date < purchase_date or estimated_date < purchase_date:
        st.warning("Delivery dates must not be before the purchase date.")
        st.stop()

    values = dict(
        purchase_date=purchase_date,
        estimated_date=estimated_date,
        delivered_date=delivered_date,
        price=float(price),
        freight_value=float(freight_value),
        payment_value=float(payment_value),
        payment_installments=int(payment_installments),
        payment_sequential=int(payment_sequential),
        product_name_length=float(name_len),
        product_description_length=float(desc_len),
        product_photos_qty=float(photos),
        product_weight_g=float(weight),
        product_length_cm=float(length_cm),
        product_height_cm=float(height_cm),
        product_width_cm=float(width_cm),
    )
    row = build_row(values, columns, dtypes)

    try:
        score = min(5.0, max(1.0, predict(endpoint, row)))
    except requests.exceptions.ConnectionError:
        st.error(
            f"Cannot reach the model server at {endpoint}. "
            "Run `python run_deployment.py --config deploy` first."
        )
        st.stop()
    except requests.exceptions.HTTPError as e:
        st.error(f"The server rejected the request: {e.response.text[:500]}")
        st.stop()

    delay = (delivered_date - estimated_date).days
    st.success(f"Predicted review score: **{score:.2f} / 5**  {'⭐' * round(score)}")
    if delay > 0:
        st.write(f"The order arrived **{delay} day(s) late** versus the estimate.")
    elif delay < 0:
        st.write(f"The order arrived **{-delay} day(s) early** versus the estimate.")
    else:
        st.write("The order arrived on the estimated date.")
    with st.expander("Features sent to the model"):
        st.dataframe(row.T.rename(columns={0: "value"}))

with st.expander("About this model"):
    st.markdown(
        """
Review scores are noisy (most are 5 stars), so predictions are rough.
On a held-out test set split **by order** (to avoid leakage from multi-item
orders), R² reaches about 0.24.

| Feature set | Best R² |
|---|---|
| Original order and product fields | 0.03 |
| + purchase-time features | 0.05 |
| + actual delivery performance | 0.23 (0.24 tuned) |

Delivery performance is the main driver, so this app predicts satisfaction
once an order has been delivered, not before it is placed.
        """
    )