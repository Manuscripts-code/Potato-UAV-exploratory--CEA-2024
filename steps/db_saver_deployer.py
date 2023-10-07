import logging

import numpy as np
from zenml import step
from zenml.integrations.mlflow.services import MLFlowDeploymentService

from configs import configs
from data_structures.schemas import Prediction, StructuredData
from database.db import SQLiteDatabase
from database.service import DBService, RecordAttributes


@step(enable_cache=False)
def db_saver_deployer(
    model_service: MLFlowDeploymentService,
    data_train: StructuredData,
    data_test: StructuredData,
    predictions_train: np.ndarray,
    predictions_test: np.ndarray,
) -> None:
    logging.info("Saving data to database...")
    deployer_cfg = model_service.config
    predictions_train = Prediction(predictions=predictions_train)
    predictions_test = Prediction(predictions=predictions_test)

    db = SQLiteDatabase()
    db_service = DBService(database=db)
    record_attrs = RecordAttributes(
        data_train=data_train,
        data_test=data_test,
        predictions_train=predictions_train,
        predictions_test=predictions_test,
    )

    db_service.update_record(
        model_name=deployer_cfg.registry_model_name,
        model_version=deployer_cfg.registry_model_version,
        record_attrs=record_attrs,
    )
