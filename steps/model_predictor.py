import logging

import numpy as np
from typing_extensions import Annotated
from zenml import step
from zenml.services import BaseService

from data_manager.structure import StructuredData


@step
def model_predictor(
    service: BaseService,
    data: StructuredData,
) -> None:
    """Run a inference request against a prediction service."""
    service.start(timeout=100)  # should be a NOP if already started
    prediction = service.predict(data.data.to_numpy())
    logging.info("Prediction: ", prediction)