import logging
from pathlib import Path

import mlflow
import numpy as np
from optuna.trial import FrozenTrial
from sklearn.pipeline import Pipeline
from zenml import step
from zenml.client import Client
from zenml.integrations.mlflow.services import MLFlowDeploymentService

from configs import configs
from configs.parser import RegistryConfig
from data_structures.schemas import Prediction, StructuredData
from database.db import SQLiteDatabase
from database.service import DBService, RecordAttributes


@step(enable_cache=False, experiment_tracker=Client().active_stack.experiment_tracker.name)
def db_saver_register(registr_cfg: RegistryConfig) -> None:
    logging.info("Saving data to database...")

    db = SQLiteDatabase()
    db_service = DBService(database=db)

    artifacts_path = Path(mlflow.active_run().info.artifact_uri)
    # record_attrs = RecordAttributes(
    #     data_train=data_train,
    #     data_test=data_test,
    #     predictions_train=predictions_train,
    #     predictions_test=predictions_test,
    # )

    # db_service.save_record(
    #     model_name=deployer_cfg.registry_model_name,
    #     model_version=deployer_cfg.registry_model_version,
    #     record_attrs=record_attrs,
    # )
