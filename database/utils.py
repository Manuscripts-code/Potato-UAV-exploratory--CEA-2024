import numpy as np

from configs import configs
from data_structures.schemas import Prediction, StructuredData
from database import schemas


def prepare_record_table(
    model_name: str,
    model_version: str,
    data_train: StructuredData,
    data_test: StructuredData,
    predictions_train: Prediction,
    predictions_test: Prediction,
):
    data_train_table = schemas.Data(name=configs.DB_DATA_TRAIN, data=data_train.to_bytes())
    data_test_table = schemas.Data(name=configs.DB_DATA_TEST, data=data_test.to_bytes())
    predictions_train_table = schemas.Prediction(
        name=configs.DB_PREDICTIONS_TRAIN, predictions=predictions_train.to_bytes()
    )
    predictions_test_table = schemas.Prediction(
        name=configs.DB_PREDICTIONS_TEST, predictions=predictions_test.to_bytes()
    )
    record_table = schemas.Record(
        model_name=model_name,
        model_version=model_version,
        data=[data_train_table, data_test_table],
        predictions=[predictions_train_table, predictions_test_table],
    )
    return record_table
