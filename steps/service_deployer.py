import logging

from typing_extensions import Annotated
from zenml.client import Client
from zenml.integrations.mlflow.services import MLFlowDeploymentService
from zenml.integrations.mlflow.steps.mlflow_deployer import mlflow_model_registry_deployer_step

from configs.parser import RegistryConfig


def service_deployer(
    registry_cfg: RegistryConfig,
) -> Annotated[MLFlowDeploymentService, "model_service"]:
    model_versions = Client().active_stack.model_registry.list_model_versions(
        name=registry_cfg.model_name,
        metadata={},
    )
    model_service = mlflow_model_registry_deployer_step.with_options(
        parameters=dict(
            registry_model_name=registry_cfg.model_name,
            registry_model_version=str(len(model_versions)),  # take the latest version
            timeout=registry_cfg.timeout,
            # or you can use the model stage if you have set it in the MLflow registry
            # registered_model_stage="None" # "Staging", "Production", "Archived"
        )
    )()
    logging.info(f"Model: {registry_cfg.model_name}, version: {len(model_versions)} will be deployed.")
    return model_service
