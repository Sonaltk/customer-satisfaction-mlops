from steps.clean_data import clean_data
from steps.evaluation import evaluation
from steps.ingest_data import ingest_data
from steps.model_train import train_model
from zenml import pipeline


@pipeline(enable_cache=False)
def train_pipeline(feature_set: str = "baseline", model_name: str = "lightgbm"):
    df = ingest_data()
    x_train, x_test, y_train, y_test = clean_data(df, feature_set=feature_set)
    model = train_model(x_train, x_test, y_train, y_test, model_name=model_name)
    evaluation(model, x_test, y_test)