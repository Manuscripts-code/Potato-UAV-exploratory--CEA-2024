import logging

from zenml import step
from zenml.integrations.mlflow.services import MLFlowDeploymentService

from configs.parser import RegistryConfig
from data_manager.structure import StructuredData


@step
def service_predictor(
    model_service: MLFlowDeploymentService,
    data: StructuredData,
    registry_cfg: RegistryConfig,
) -> None:
    """Run a inference request against a prediction service."""
    model_service.start(timeout=registry_cfg.timeout)  # should be a NOP if already started
    prediction = model_service.predict(data.data.to_numpy())
    logging.info(f"Prediction: {prediction}")
