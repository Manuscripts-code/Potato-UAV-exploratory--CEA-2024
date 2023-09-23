from dataclasses import dataclass

import pandas as pd
from rich import print

from data_structures.schemas import ClassificationTarget, RegressionTarget
from database.db import SQLiteDatabase
from utils.metrics import calculate_classification_metrics, calculate_regression_metrics


@dataclass(frozen=True, slots=True)
class Column:
    model_name = ("model", "name")
    model_version = ("model", "version")
    model_is_latest = ("model", "is_latest")
    model_data_name = ("model", "data_name")
    model_prediction_name = ("model", "prediction_name")

    def to_list(self):
        # get all attributes values of the class and remove this method
        attributes = dir(self)
        attributes.remove("to_list")
        return [getattr(self, attr) for attr in attributes if not attr.startswith("__")]


@dataclass(frozen=True, slots=True)
class ClassificationColumn(Column):
    metrics_classification_accuracy = ("metrics_clf", "accuracy")
    metrics_classification_precision = ("metrics_clf", "precision")
    metrics_classification_recall = ("metrics_clf", "recall")
    metrics_classification_f1 = ("metrics_clf", "f1")
    metrics_classification_roc_auc = ("metrics_clf", "roc_auc")


@dataclass(frozen=True, slots=True)
class RegressionColumn(Column):
    metrics_regression_mae = ("metrics_reg", "mae")
    metrics_regression_mse = ("metrics_reg", "mse")
    metrics_regression_rmse = ("metrics_reg", "rmse")
    metrics_regression_r2 = ("metrics_reg", "r2")


Column = Column()
RegressionColumn = RegressionColumn()
ClassificationColumn = ClassificationColumn()


class Report:
    def __init__(self):
        self._df_clf = pd.DataFrame(columns=pd.MultiIndex.from_tuples(ClassificationColumn.to_list()))
        self._df_reg = pd.DataFrame(columns=pd.MultiIndex.from_tuples(RegressionColumn.to_list()))

    def add_record(
        self,
        model_name,
        model_version,
        model_is_latest,
        data_name,
        data_content,
        pred_name,
        pred_content,
    ):
        data = data_content.data
        meta = data_content.meta
        target = data_content.target

        y_true = target.value.to_numpy()
        y_pred = pred_content.predictions

        model_columns = {
            Column.model_name: model_name,
            Column.model_version: model_version,
            Column.model_is_latest: model_is_latest,
            Column.model_data_name: data_name,
            Column.model_prediction_name: pred_name,
        }

        if isinstance(target, ClassificationTarget):
            metrics = calculate_classification_metrics(y_true, y_pred)
            self._add_classification_record(model_columns, metrics)
        elif isinstance(target, RegressionTarget):
            metrics = calculate_regression_metrics(y_true, y_pred)
            self._add_regression_record(model_columns, metrics)
        else:
            raise ValueError("Unknown target type")

    def _add_classification_record(self, model_columns, metrics):
        metrics_columns = {
            ClassificationColumn.metrics_classification_accuracy: [metrics.accuracy],
            ClassificationColumn.metrics_classification_precision: [metrics.precision],
            ClassificationColumn.metrics_classification_recall: [metrics.recall],
            ClassificationColumn.metrics_classification_f1: [metrics.f1],
            ClassificationColumn.metrics_classification_roc_auc: [metrics.roc_auc],
        }
        columns = {**model_columns, **metrics_columns}
        self._df_clf = pd.concat([self._df_clf, pd.DataFrame.from_dict(columns)], ignore_index=True)

    def _add_regression_record(self, model_columns, metrics):
        metrics_columns = {
            RegressionColumn.metrics_regression_mae: [metrics.mae],
            RegressionColumn.metrics_regression_mse: [metrics.mse],
            RegressionColumn.metrics_regression_rmse: [metrics.rmse],
            RegressionColumn.metrics_regression_r2: [metrics.r2],
        }
        columns = {**model_columns, **metrics_columns}
        self._df_reg = pd.concat([self._df_reg, pd.DataFrame.from_dict(columns)], ignore_index=True)


if __name__ == "__main__":
    db = SQLiteDatabase()
    report = Report()

    records = db.get_all_records()
    records_latest = db.get_latest_records()

    for record in records:
        model_name = record.model_name
        model_version = record.model_version
        model_is_latest = record.is_latest

        for data, predictions in zip(record.data, record.predictions):
            data_name = data.name
            data_content = data.content
            pred_name = predictions.name
            pred_content = predictions.content
            report.add_record(
                model_name,
                model_version,
                model_is_latest,
                data_name,
                data_content,
                pred_name,
                pred_content,
            )

    pass
