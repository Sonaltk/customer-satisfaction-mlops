import click
from pipelines.training_pipeline import train_pipeline
from zenml.client import Client


@click.command()
@click.option(
    "--feature-set",
    type=click.Choice(["baseline", "pre_order", "full"]),
    default="baseline",
)
@click.option(
    "--model-name",
    type=click.Choice(["lightgbm", "xgboost", "randomforest", "linear_regression"]),
    default="lightgbm",
)
@click.option(
    "--split-by",
    type=click.Choice(["order_id", "customer_unique_id", "none"]),
    default="order_id",
)
def main(feature_set: str, model_name: str, split_by: str):
    print(Client().active_stack.experiment_tracker.get_tracking_uri())
    train_pipeline(feature_set=feature_set, model_name=model_name, split_by=split_by)


if __name__ == "__main__":
    main()