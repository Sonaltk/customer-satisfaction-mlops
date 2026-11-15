from steps.clean_data import clean_data
from steps.evaluation import evaluation
from steps.ingest_data import ingest_data
from steps.model_train import train_model
from zenml import pipeline


@pipeline(enable_cache=False)
def train_pipeline(
    feature_set: str = "baseline",
    model_name: str = "lightgbm",
    split_by: str = "order_id",
    fine_tuning: bool = False,
):
    df = ingest_data()
    x_train, x_test, y_train, y_test, groups_train = clean_data(
        df, feature_set=feature_set, split_by=split_by
    )
    model = train_model(
        x_train,
        y_train,
        groups_train,
        model_name=model_name,
        fine_tuning=fine_tuning,
        feature_set=feature_set,
        split_by=split_by,
    )
    evaluation(model, x_test, y_test)