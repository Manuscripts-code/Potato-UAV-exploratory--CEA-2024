import logging

import numpy as np
from zenml import step
from zenml.integrations.mlflow.services import MLFlowDeploymentService

from configs import configs
from data_structures.schemas import Prediction, StructuredData
from database.db import SQLiteDatabase
from database.utils import prepare_record_table


@step(enable_cache=configs.CACHING)
def db_saver(
    model_service: MLFlowDeploymentService,
    data_train: StructuredData,
    data_test: StructuredData,
    predictions_train: np.ndarray,
    predictions_test: np.ndarray,
) -> None:
    logging.info("Saving data to database...")
    predictions_train = Prediction(predictions=predictions_train, name=configs.DB_PREDICTIONS_TRAIN)
    predictions_test = Prediction(predictions=predictions_test, name=configs.DB_PREDICTIONS_TEST)
    record_table = prepare_record_table(
        "iris",
        "v1",
        data_train,
        data_test,
        predictions_train,
        predictions_test,
    )
    db = SQLiteDatabase()
    db.save_record(record_table)
