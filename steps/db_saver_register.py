import logging

import mlflow
from zenml import step
from zenml.client import Client
from zenml.utils import dashboard_utils

from configs.parser import RegistryConfig
from database.db import SQLiteDatabase
from database.service import DBService, RecordAttributes


@step(enable_cache=False, experiment_tracker=Client().active_stack.experiment_tracker.name)
def db_saver_register(registr_cfg: RegistryConfig) -> None:
    logging.info("Saving data to database...")

    runs = Client().list_pipeline_runs(
        sort_by="desc:start_time",
        size=1,
    )

    record_attrs = RecordAttributes(
        mlflow_uri=mlflow.active_run().info.artifact_uri,
        dashboard_url=dashboard_utils.get_run_url(runs[0]),
    )

    db = SQLiteDatabase()
    db_service = DBService(database=db)
    db_service.create_record(
        model_name=registr_cfg.model_name,
        record_attrs=record_attrs,
    )
