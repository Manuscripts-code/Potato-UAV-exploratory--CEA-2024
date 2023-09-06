import logging

from zenml import step
from zenml.integrations.mlflow.services import MLFlowDeploymentService

from data_manager.structure import StructuredData


@step
def service_predictor(
    model_service: MLFlowDeploymentService,
    data: StructuredData,
) -> None:
    """Run a inference request against a prediction service."""
    model_service.start(timeout=100)  # should be a NOP if already started
    prediction = model_service.predict(data.data.to_numpy())
    logging.info(f"Prediction: {prediction}")
