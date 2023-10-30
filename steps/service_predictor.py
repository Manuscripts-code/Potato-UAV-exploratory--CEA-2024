import logging

import numpy as np
from typing_extensions import Annotated
from zenml import step
from zenml.integrations.mlflow.services import MLFlowDeploymentService

from configs import configs
from configs.parser import RegistryConfig
from data_structures.schemas import StructuredData


@step
def service_predictor(
    model_service: MLFlowDeploymentService,
    data: StructuredData,
    registry_cfg: RegistryConfig,
) -> Annotated[np.ndarray, "predictions"]:
    """Run a inference request against a prediction service."""
    model_service.start(timeout=registry_cfg.timeout)  # should be a NOP if already started
    return model_service.predict(data.data)
