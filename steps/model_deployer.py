from typing_extensions import Annotated
from zenml.integrations.mlflow.services import MLFlowDeploymentService
from zenml.integrations.mlflow.steps.mlflow_deployer import mlflow_model_registry_deployer_step

from configs.parser import RegistryConfig


def model_deployer(registry_cfg: RegistryConfig) -> Annotated[MLFlowDeploymentService, "model_service"]:
    model_service = mlflow_model_registry_deployer_step.with_options(
        parameters=dict(
            registry_model_name=registry_cfg.model_name,
            registry_model_version="1",
            timeout=300,
            # or you can use the model stage if you have set it in the MLflow registry
            # registered_model_stage="None" # "Staging", "Production", "Archived"
        )
    )()
    return model_service
