import json

import numpy as np
import pandas as pd
from steps.clean_data import clean_data
from steps.evaluation import evaluation
from steps.ingest_data import ingest_data
from steps.model_train import train_model
from zenml import pipeline, step
from zenml.config import DockerSettings
from zenml.constants import DEFAULT_SERVICE_START_STOP_TIMEOUT
from zenml.integrations.constants import MLFLOW
from zenml.integrations.mlflow.model_deployers.mlflow_model_deployer import (
    MLFlowModelDeployer,
)
from zenml.integrations.mlflow.services import MLFlowDeploymentService
from zenml.integrations.mlflow.steps import mlflow_model_deployer_step

from .utils import get_data_for_test

docker_settings = DockerSettings(required_integrations=[MLFLOW])


@step(enable_cache=False)
def dynamic_importer(feature_set: str) -> str:
    """Builds a small batch of inference data, preprocessed like the training data."""
    return get_data_for_test(feature_set=feature_set)


@step
def deployment_trigger(mse: float, max_mse: float = 1.6) -> bool:
    """Deploy only if the test-set MSE is at or below the threshold (lower is better)."""
    return mse <= max_mse


@step(enable_cache=False)
def prediction_service_loader(
    pipeline_name: str,
    pipeline_step_name: str,
    running: bool = True,
    model_name: str = "model",
) -> MLFlowDeploymentService:
    """Get the prediction service started by the deployment pipeline."""
    model_deployer = MLFlowModelDeployer.get_active_model_deployer()
    existing_services = model_deployer.find_model_server(
        pipeline_name=pipeline_name,
        pipeline_step_name=pipeline_step_name,
        model_name=model_name,
        running=running,
    )
    if not existing_services:
        raise RuntimeError(
            f"No MLflow prediction service deployed by the {pipeline_step_name} "
            f"step in the {pipeline_name} pipeline for the '{model_name}' model "
            f"is currently running."
        )
    return existing_services[0]


@step
def predictor(service: MLFlowDeploymentService, data: str) -> np.ndarray:
    """Run an inference request against a prediction service."""
    service.start(timeout=10)  # no-op if already running
    payload = json.loads(data)
    df = pd.DataFrame(payload["data"], columns=payload["columns"])
    records = np.array(json.loads(df.to_json(orient="records")))
    return service.predict(records)


@pipeline(enable_cache=False, settings={"docker": docker_settings})
def continuous_deployment_pipeline(
    feature_set: str = "full",
    model_name: str = "lightgbm",
    split_by: str = "order_id",
    fine_tuning: bool = False,
    max_mse: float = 1.6,
    workers: int = 1,
    timeout: int = DEFAULT_SERVICE_START_STOP_TIMEOUT,
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
    mse, r2_score, rmse = evaluation(model, x_test, y_test)
    deployment_decision = deployment_trigger(mse=mse, max_mse=max_mse)
    mlflow_model_deployer_step(
        model=model,
        deploy_decision=deployment_decision,
        workers=workers,
        timeout=timeout,
    )


@pipeline(enable_cache=False, settings={"docker": docker_settings})
def inference_pipeline(
    pipeline_name: str, pipeline_step_name: str, feature_set: str = "full"
):
    batch_data = dynamic_importer(feature_set=feature_set)
    service = prediction_service_loader(
        pipeline_name=pipeline_name,
        pipeline_step_name=pipeline_step_name,
        running=False,
    )
    predictor(service=service, data=batch_data)