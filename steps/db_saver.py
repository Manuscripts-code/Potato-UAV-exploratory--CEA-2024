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
    deployer_cfg = model_service.config
    predictions_train = Prediction(predictions=predictions_train)
    predictions_test = Prediction(predictions=predictions_test)

    db = SQLiteDatabase()

    logging.info(
        f"Record for model name: '{deployer_cfg.registry_model_name}' with version: "
        f"'{deployer_cfg.registry_model_version}' is being checked..."
    )
    if db.get_record(
        model_name=deployer_cfg.registry_model_name, model_version=deployer_cfg.registry_model_version
    ):
        logging.warning("Record already exists. Skipping...")
        return

    logging.info("Record does not exist. Saving record to database...")
    record_table = prepare_record_table(
        model_name=deployer_cfg.registry_model_name,
        model_version=deployer_cfg.registry_model_version,
        data_train=data_train,
        data_test=data_test,
        predictions_train=predictions_train,
        predictions_test=predictions_test,
    )
    db.reset_latest_record(model_name=deployer_cfg.registry_model_name)
    db.save_record(record_table)
