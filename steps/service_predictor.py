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
    # ! To use the model_service.predict method, the registered model needed to be fitted on data in numpy format.
    return model_service.predict(data.data.to_numpy())
