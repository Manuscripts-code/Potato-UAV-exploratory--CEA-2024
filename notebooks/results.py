import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import pandas as pd
from rich import print
from sklearn.metrics import classification_report
from ydata_profiling import ProfileReport

sys.path.insert(0, "..")


from configs import configs
from data_structures.schemas import ClassificationTarget, Prediction, RegressionTarget, StructuredData
from database.db import SQLiteDatabase
from database.schemas import MetricSchema, RecordSchema
from utils.metrics import (
    ClassificationMetrics,
    RegressionMetrics,
    calculate_classification_metrics,
    calculate_regression_metrics,
)
from utils.plot_utils import (
    save_confusion_matrix_display,
    save_data_visualization,
    save_meta_visualization,
    save_prediction_errors_display,
    save_target_visualization,
)
from utils.utils import ensure_dir, write_txt


@dataclass(frozen=True, slots=True)
class Column:
    model_name = ("model", "name")
    model_version = ("model", "version")
    model_is_latest = ("model", "is_latest")
    model_created_at = ("model", "created_at")
    model_data_name = ("model", "data_name")
    model_cv_metric = ("model", configs.DB_CV_METRIC_NAME)

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
    metrics_regression_mape = ("metrics_reg", "mape")
    metrics_regression_maxe = ("metrics_reg", "maxe")


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
            model_mlflow_uri = record.mlflow_uri
            model_is_latest = record.is_latest
            model_created_at = record.created_at
            model_metrics = record.metrics

            for data, predictions in zip(record.data, record.predictions):
                data_name = data.name
                data_content = data.content
                pred_content = predictions.content
                metrics = self.add_record_metrics(
                    model_name=model_name,
                    model_version=model_version,
                    model_is_latest=model_is_latest,
                    model_created_at=model_created_at,
                    data_name=data_name,
                    data_content=data_content,
                    pred_content=pred_content,
                    model_metrics=model_metrics,
                )

                self.save_record_artifacts(
                    model_name=model_name,
                    model_version=model_version,
                    model_mlflow_uri=model_mlflow_uri,
                    data_name=data_name,
                    data_content=data_content,
                    pred_content=pred_content,
                    metrics=metrics,
                )

        self.save_records_summary()

    def save_records_summary(self):
        write_txt(self.df_classification.to_string(), configs.SAVE_DIR / "classification.txt")
        write_txt(self.df_regression.to_string(), configs.SAVE_DIR / "regression.txt")

    def save_record_artifacts(
        self,
        model_name: str,
        model_version: str,
        model_mlflow_uri: str,
        data_name: str,
        data_content: StructuredData,
        pred_content: Prediction,
        metrics: NamedTuple,
    ):
        data = data_content.data
        meta = data_content.meta
        target = data_content.target

        y_true = target.value.to_numpy()
        y_pred = pred_content.predictions

        save_dir = ensure_dir(Path(configs.SAVE_RESULTS_DIR, model_name, model_version, data_name))

        profile = ProfileReport(pd.concat([data.reset_index(drop=True), meta.reset_index(drop=True), target.value.to_frame().reset_index(drop=True)], axis=1), title="Report")  # type: ignore # noqa
        profile.to_file(save_dir / "profiling_report.html")

        write_txt(data.describe().to_string(), save_dir / "describe_data.txt")
        write_txt(meta.groupby([configs.TREATMENT_ENG, configs.DATE_ENG, configs.BLOCK_ENG, configs.VARIETY_ENG]).size().to_string(), save_dir / "describe_meta.txt")  # type: ignore # noqa
        write_txt(data.to_string(), save_dir / "data_data.txt")
        write_txt(meta.to_string(), save_dir / "data_meta.txt")
        write_txt(str(metrics), save_dir / "metrics.txt")
        save_meta_visualization(meta, save_dir=save_dir)
        self._copy_shap_artifacts(Path(model_mlflow_uri) / configs.MLFLOW_EXPLAINER / data_name, save_dir)  # type: ignore # noqa

        if isinstance(target, ClassificationTarget):
            if isinstance(target.label[0], list):
                row_formatter = lambda row: "__".join(row)
            else:
                row_formatter = lambda row: row

            target_label = target.label.apply(row_formatter)
            encoding = target.encoding.apply(row_formatter).to_dict()
            target_names = [encoding[key] for key in sorted(encoding.keys())]

            save_confusion_matrix_display(y_true, y_pred, target_names, save_path=save_dir / "confusion_matrix.pdf")  # type: ignore # noqa
            write_txt(classification_report(y_true, y_pred, target_names=target_names), save_dir / "classification_report.txt")  # type: ignore # noqa
            write_txt(pd.concat([target_label, target.value], axis=1).to_string(), save_dir / "data_target.txt")  # type: ignore # noqa
            save_data_visualization(data, y_data_encoded=y_true, classes=target_names, save_dir=save_dir)  # type: ignore # noqa
            save_target_visualization(target_values=y_true, target_labels=target_label.to_numpy(), save_path=save_dir / "visualization_target.pdf")  # type: ignore # noqa

        elif isinstance(target, RegressionTarget):
            target_label = meta[configs.VARIETY_ENG]
            target_label_encoded, target_label_classes = pd.factorize(target_label)
            target_label_classes = target_label_classes.tolist()

            save_prediction_errors_display(y_true, y_pred, kind="residual_vs_predicted", save_path=save_dir / "prediction_errors_rvp.pdf")  # type: ignore # noqa
            save_prediction_errors_display(y_true, y_pred, kind="actual_vs_predicted", save_path=save_dir / "prediction_errors_avp.pdf")  # type: ignore # noqa
            write_txt(target.value.to_string(), save_dir / "data_target.txt")
            save_data_visualization(data, y_data_encoded=target_label_encoded, classes=target_label_classes, save_dir=save_dir)  # type: ignore # noqa
            save_target_visualization(target_values=y_true, target_labels=target_label.to_numpy(), target_type="regression", save_path=save_dir / "visualization_target.pdf")  # type: ignore # noqa

        else:
            raise ValueError(f"Unknown target type: {type(target)}")

    def _copy_shap_artifacts(self, explainer_artifacts_uri: str, save_dir: Path):
        explainer_artifacts_uri = Path(explainer_artifacts_uri)
        explainer_artifacts_uri = Path("/", *explainer_artifacts_uri.parts[1:])
        shutil.copytree(explainer_artifacts_uri, save_dir, dirs_exist_ok=True)

    def add_record_metrics(
        self,
        model_name: str,
        model_version: str,
        model_is_latest: bool,
        model_created_at: str,
        data_name: str,
        data_content: StructuredData,
        pred_content: Prediction,
        model_metrics: list[MetricSchema],
    ) -> NamedTuple:
        target = data_content.target

        y_true = target.value.to_numpy()
        y_pred = pred_content.predictions

        assert len(model_metrics) == 1, "Only one model metric is supported"
        model_columns = {
            Column.model_name: model_name,
            Column.model_version: model_version,
            Column.model_is_latest: model_is_latest,
            Column.model_created_at: model_created_at,
            Column.model_data_name: data_name,
            Column.model_cv_metric: model_metrics[0].value,
        }

        if isinstance(target, ClassificationTarget):
            metrics = calculate_classification_metrics(y_true, y_pred)
            self._add_classification_record(model_columns, metrics)
        elif isinstance(target, RegressionTarget):
            metrics = calculate_regression_metrics(y_true, y_pred)
            self._add_regression_record(model_columns, metrics)
        else:
            raise ValueError(f"Unknown target type: {type(target)}")
        return metrics

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
            RegressionColumn.metrics_regression_mape: [metrics.mape],
            RegressionColumn.metrics_regression_maxe: [metrics.maxe],
        }
        columns = {**model_columns, **metrics_columns}
        self._df_reg = pd.concat([self._df_reg, pd.DataFrame.from_dict(columns)], ignore_index=True)

    @property
    def df_classification(self):
        return self._df_clf

    @property
    def df_regression(self):
        return self._df_reg


def produce_results(number_of_recents: int | None = None):
    db = SQLiteDatabase()

    records = db.get_records()
    records_latest = db.get_records(is_latest=True)

    if number_of_recents is not None:
        records = records[-number_of_recents:]
        records_latest = records_latest[-number_of_recents:]

    # report = Report()
    # report.add_records(records)
    # print(report.df_classification)
    # print(report.df_regression)

    report = Report()
    report.add_records(records_latest)
    print(report.df_classification)
    print(report.df_regression)


if __name__ == "__main__":
    number_of_recents = 1
    # number_of_recents = None
    produce_results(number_of_recents)
