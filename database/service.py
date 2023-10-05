import logging
from dataclasses import dataclass

import numpy as np

from configs import configs
from data_structures.schemas import Prediction, StructuredData
from database import schemas
from database.db import SQLiteDatabase


@dataclass
class RecordAttributes:
    data_train: StructuredData = None
    data_test: StructuredData = None
    predictions_train: Prediction = None
    predictions_test: Prediction = None


class DBService:
    def __init__(self, database: SQLiteDatabase):
        self.db = database

    def prepare_record_table(
        self,
        model_name: str,
        model_version: str,
        data_train: StructuredData,
        data_test: StructuredData,
        predictions_train: Prediction,
        predictions_test: Prediction,
    ):
        data_train_table = schemas.DataSchema(name=configs.DB_DATA_TRAIN, content=data_train.to_bytes())
        data_test_table = schemas.DataSchema(name=configs.DB_DATA_TEST, content=data_test.to_bytes())
        predictions_train_table = schemas.PredictionSchema(
            name=configs.DB_PREDICTIONS_TRAIN, content=predictions_train.to_bytes()
        )
        predictions_test_table = schemas.PredictionSchema(
            name=configs.DB_PREDICTIONS_TEST, content=predictions_test.to_bytes()
        )
        record_table = schemas.RecordSchema(
            model_name=model_name,
            model_version=model_version,
            data=[data_train_table, data_test_table],
            predictions=[predictions_train_table, predictions_test_table],
        )
        return record_table

    def save_record(self, model_name: str, model_version: str, record_attrs: RecordAttributes):
        if not isinstance(record_attrs, RecordAttributes):
            logging.error("Record attributes are not of type RecordAttributes.")
            return

        logging.info(
            f"Record for model name: '{model_name}' with version: "
            f"'{model_version}' is being checked..."
        )
        if self.db.get_record(
            model_name=model_name,
            model_version=model_version,
        ):
            logging.warning("Record already exists. Skipping...")
            return
        logging.info("Record does not exist. Saving record to database...")

        record_table = self.prepare_record_table(
            model_name=model_name,
            model_version=model_version,
            data_train=record_attrs.data_train,
            data_test=record_attrs.data_test,
            predictions_train=record_attrs.predictions_train,
            predictions_test=record_attrs.predictions_test,
        )
        self.db.reset_latest_record(model_name=model_name)
        self.db.save_record(record_table)
