import logging
from dataclasses import dataclass
from functools import wraps

import numpy as np

from configs import configs
from data_structures.schemas import Prediction, StructuredData
from database import schemas
from database.db import SQLiteDatabase


@dataclass(frozen=True, slots=True)
class RecordAttributes:
    mlflow_uri: str = None
    dashboard_url: str = None
    data_train: StructuredData = None
    data_test: StructuredData = None
    predictions_train: Prediction = None
    predictions_test: Prediction = None


class DBService:
    def __init__(self, database: SQLiteDatabase):
        self.db = database

    def record_check(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            if not isinstance(kwargs["record_attrs"], RecordAttributes):
                logging.error("Record attributes are not of type RecordAttributes.")
                raise TypeError

            logging.info(f"Record for model name: '{kwargs['model_name']}'")
            return func(self, *args, **kwargs)

        return wrapper

    @record_check
    def create_record(self, model_name: str, model_version: str, record_attrs: RecordAttributes):
        logging.info("Record does not exist. Saving record to database...")
        records = self.db.get_records(model_name=model_name)
        if records:
            self.db.update_record(model_name=model_name, to_update={"is_latest": False}, is_latest=True)
        record_table = self._create_record_table(
            model_name=model_name, model_version=model_version, record_attrs=record_attrs
        )
        self.db.save_record(record_table)
        logging.info(f"Record with model version {model_version} saved to database.")

    @record_check
    def update_record(self, model_name: str, model_version: str, record_attrs: RecordAttributes):
        record_update = self._update_record_table(record_attrs=record_attrs)
        self.db.update_record(
            model_name=model_name, model_version=model_version, to_update=record_update
        )
        logging.info(f"Record with model version {model_version} updated.")

    def _create_record_table(
        self, model_name: str, model_version: str, record_attrs: RecordAttributes
    ) -> schemas.RecordSchema:
        record_table = schemas.RecordSchema(
            model_name=model_name,
            model_version=model_version,
            mlflow_uri=record_attrs.mlflow_uri,
            dashboard_url=record_attrs.dashboard_url,
        )
        return record_table

    def _update_record_table(self, record_attrs: RecordAttributes):
        data_train_table = schemas.DataSchema(
            name=configs.DB_DATA_TRAIN, content=record_attrs.data_train.to_bytes()
        )
        data_test_table = schemas.DataSchema(
            name=configs.DB_DATA_TEST, content=record_attrs.data_test.to_bytes()
        )
        predictions_train_table = schemas.PredictionSchema(
            name=configs.DB_PREDICTIONS_TRAIN, content=record_attrs.predictions_train.to_bytes()
        )
        predictions_test_table = schemas.PredictionSchema(
            name=configs.DB_PREDICTIONS_TEST, content=record_attrs.predictions_test.to_bytes()
        )
        _data = "data"
        _predictions = "predictions"
        if not hasattr(schemas.RecordSchema, _data) or not hasattr(schemas.RecordSchema, _predictions):
            logging.error(f"RecordSchema does not have '{_data}' or '{_predictions}' attribute.")
            raise AttributeError

        record_update = {
            _data: [data_train_table, data_test_table],
            _predictions: [predictions_train_table, predictions_test_table],
        }
        return record_update
