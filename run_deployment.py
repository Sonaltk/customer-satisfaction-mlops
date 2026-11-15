from typing import cast

import click
from pipelines.deployment_pipeline import (
    continuous_deployment_pipeline,
    inference_pipeline,
)
from rich import print
from zenml.integrations.mlflow.mlflow_utils import get_tracking_uri
from zenml.integrations.mlflow.model_deployers.mlflow_model_deployer import (
    MLFlowModelDeployer,
)
from zenml.integrations.mlflow.services import MLFlowDeploymentService

DEPLOY = "deploy"
PREDICT = "predict"
DEPLOY_AND_PREDICT = "deploy_and_predict"


@click.command()
@click.option(
    "--config",
    "-c",
    type=click.Choice([DEPLOY, PREDICT, DEPLOY_AND_PREDICT]),
    default=DEPLOY_AND_PREDICT,
    help="`deploy` trains and (maybe) deploys, `predict` queries the running "
    "service, `deploy_and_predict` does both.",
)
@click.option(
    "--feature-set",
    type=click.Choice(["baseline", "pre_order", "full"]),
    default="full",
)
@click.option(
    "--model-name",
    type=click.Choice(["lightgbm", "xgboost", "randomforest", "linear_regression"]),
    default="lightgbm",
)
@click.option("--fine-tuning", is_flag=True, help="Tune hyperparameters with Optuna.")
@click.option(
    "--max-mse",
    default=1.6,
    help="Deploy only if the test MSE is at or below this value (lower is better).",
)
def main(config: str, feature_set: str, model_name: str, fine_tuning: bool, max_mse: float):
    model_deployer = MLFlowModelDeployer.get_active_model_deployer()
    deploy = config in (DEPLOY, DEPLOY_AND_PREDICT)
    predict = config in (PREDICT, DEPLOY_AND_PREDICT)

    if deploy:
        continuous_deployment_pipeline(
            feature_set=feature_set,
            model_name=model_name,
            fine_tuning=fine_tuning,
            max_mse=max_mse,
            workers=3,
            timeout=60,
        )

    if predict:
        inference_pipeline(
            pipeline_name="continuous_deployment_pipeline",
            pipeline_step_name="mlflow_model_deployer_step",
            feature_set=feature_set,
        )

    print(
        "Inspect your runs with:\n"
        f"[italic green]    mlflow ui --backend-store-uri '{get_tracking_uri()}'[/italic green]\n"
    )

    existing_services = model_deployer.find_model_server(
        pipeline_name="continuous_deployment_pipeline",
        pipeline_step_name="mlflow_model_deployer_step",
        model_name="model",
    )

    if existing_services:
        service = cast(MLFlowDeploymentService, existing_services[0])
        if service.is_running:
            print(
                "The MLflow prediction server is running locally and accepts "
                f"requests at:\n    {service.prediction_url}\n"
                "To stop it, run "
                f"[italic green]`zenml model-deployer models delete {str(service.uuid)}`[/italic green]."
            )
        elif service.is_failed:
            print(
                "The MLflow prediction server is in a failed state:\n"
                f" Last state: '{service.status.state.value}'\n"
                f" Last error: '{service.status.last_error}'"
            )
    else:
        print(
            "No MLflow prediction server is currently running. Either the "
            "deployment pipeline has not run yet, or the model did not pass "
            "the MSE gate."
        )


if __name__ == "__main__":
    main()