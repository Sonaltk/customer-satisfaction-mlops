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
def main(feature_set: str, model_name: str):
    print(Client().active_stack.experiment_tracker.get_tracking_uri())
    train_pipeline(feature_set=feature_set, model_name=model_name)


if __name__ == "__main__":
    main()