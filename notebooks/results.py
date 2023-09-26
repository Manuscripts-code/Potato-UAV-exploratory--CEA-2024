import sys

from sklearn.metrics import classification_report

sys.path.insert(0, "..")

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from rich import print

from configs import configs
from data_structures.schemas import ClassificationTarget, Prediction, RegressionTarget, StructuredData
from database.db import SQLiteDatabase
from database.schemas import RecordSchema
from utils.metrics import (
    ClassificationMetrics,
    RegressionMetrics,
    calculate_classification_metrics,
    calculate_regression_metrics,
)
from utils.plot_utils import save_confusion_matrix, save_features_plot
from utils.utils import ensure_dir, write_txt


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

    def add_records(self, records: RecordSchema):
        for record in records:
            model_name = record.model_name
            model_version = record.model_version
            model_is_latest = record.is_latest

            for data, predictions in zip(record.data, record.predictions):
                data_name = data.name
                data_content = data.content
                pred_name = predictions.name
                pred_content = predictions.content
                self.add_record(
                    model_name=model_name,
                    model_version=model_version,
                    model_is_latest=model_is_latest,
                    data_name=data_name,
                    data_content=data_content,
                    pred_name=pred_name,
                    pred_content=pred_content,
                )

                self.save_record_artifacts(
                    model_name=model_name,
                    model_version=model_version,
                    data_name=data_name,
                    data_content=data_content,
                    pred_content=pred_content,
                )

    def save_record_artifacts(
        self,
        model_name: str,
        model_version: str,
        data_name: str,
        data_content: StructuredData,
        pred_content: Prediction,
    ):
        data = data_content.data
        meta = data_content.meta
        target = data_content.target

        y_true = target.value.to_numpy()
        y_pred = pred_content.predictions

        save_dir = ensure_dir(Path(configs.SAVE_RESULTS_DIR, model_name, model_version, data_name))

        write_txt(data.describe().to_string(), save_dir / "describe_data.txt")
        write_txt(data.to_string(), save_dir / "data_data.txt")
        write_txt(meta.to_string(), save_dir / "data_meta.txt")

        if isinstance(target, ClassificationTarget):
            row_formatter = lambda row: "__".join(row)
            target_label = target.label.apply(row_formatter)
            encoding = target.encoding.apply(row_formatter).to_dict()
            target_names = [encoding[key] for key in sorted(encoding.keys())]

            save_features_plot(data, data.columns.tolist(), target_label.to_numpy(), save_path=save_dir / "features_plot.pdf")  # type: ignore # noqa
            save_confusion_matrix(y_true, y_pred, target_names, save_path=save_dir / "confusion_matrix.pdf")  # type: ignore # noqa
            write_txt(classification_report(y_true, y_pred, target_names=target_names), save_dir / "classification_report.txt")  # type: ignore # noqa
            write_txt(pd.concat([target_label, target.value], axis=1).to_string(), save_dir / "data_target.txt")  # type: ignore # noqa

        elif isinstance(target, RegressionTarget):
            write_txt(target.value.to_string(), save_dir / "data_target.txt")  # type: ignore # noqa

        else:
            raise ValueError(f"Unknown target type: {type(target)}")

    def add_record(
        self,
        model_name: str,
        model_version: str,
        model_is_latest: bool,
        data_name: str,
        data_content: StructuredData,
        pred_name: str,
        pred_content: Prediction,
    ):
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
            raise ValueError(f"Unknown target type: {type(target)}")

    def _add_classification_record(
        self,
        model_columns: dict[tuple[str, str], any],
        metrics: ClassificationMetrics | RegressionMetrics,
    ):
        metrics_columns = {
            ClassificationColumn.metrics_classification_accuracy: [metrics.accuracy],
            ClassificationColumn.metrics_classification_precision: [metrics.precision],
            ClassificationColumn.metrics_classification_recall: [metrics.recall],
            ClassificationColumn.metrics_classification_f1: [metrics.f1],
        }
        columns = {**model_columns, **metrics_columns}
        self._df_clf = pd.concat([self._df_clf, pd.DataFrame.from_dict(columns)], ignore_index=True)

    def _add_regression_record(
        self,
        model_columns: dict[tuple[str, str], any],
        metrics: ClassificationMetrics | RegressionMetrics,
    ):
        metrics_columns = {
            RegressionColumn.metrics_regression_mae: [metrics.mae],
            RegressionColumn.metrics_regression_mse: [metrics.mse],
            RegressionColumn.metrics_regression_rmse: [metrics.rmse],
            RegressionColumn.metrics_regression_r2: [metrics.r2],
        }
        columns = {**model_columns, **metrics_columns}
        self._df_reg = pd.concat([self._df_reg, pd.DataFrame.from_dict(columns)], ignore_index=True)

    @property
    def df_classification(self):
        return self._df_clf

    @property
    def df_regression(self):
        return self._df_reg


if __name__ == "__main__":
    db = SQLiteDatabase()

    records = db.get_all_records()
    records_latest = db.get_latest_records()

    # report = Report()
    # report.add_records(records)
    # print(report.df_classification)
    # print(report.df_regression)

    report = Report()
    report.add_records(records_latest)
    print(report.df_classification)
    print(report.df_regression)
