import logging
import tempfile
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import joblib
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import shap
from optuna.trial import FrozenTrial
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline

from configs import configs
from data_structures.schemas import ClassificationTarget, StructuredData
from utils.metrics import calculate_classification_metrics, calculate_regression_metrics
from utils.plot_utils import save_plot_figure
from utils.utils import ensure_dir, replace_substring, write_json, write_txt


@dataclass
class TransferObject:
    best_model: Pipeline
    best_trial: FrozenTrial
    y_pred: np.ndarray
    y_true: np.ndarray
    label: np.ndarray
    encoding: dict
    data: pd.DataFrame
    meta: pd.DataFrame
    suffix: str


class ArtifactLogger(Protocol):
    def log_params(self, tobj: TransferObject):
        ...

    def log_metrics(self, tobj: TransferObject):
        ...

    def log_artifacts(self, tobj: TransferObject):
        ...


class Evaluator:
    def __init__(
        self,
        best_model: Pipeline,
        best_trial: FrozenTrial,
        logger: ArtifactLogger,
    ):
        self.best_model = deepcopy(best_model)
        self.best_trial = best_trial
        self.logger = logger

    def run(self, data: StructuredData, suffix: str):
        if isinstance(data.target, ClassificationTarget):
            label = data.target.label.to_numpy()
            encoding = data.target.encoding.to_dict()
        else:
            label = None
            encoding = None

        transfer_object = TransferObject(
            best_model=self.best_model,
            best_trial=self.best_trial,
            y_pred=self.best_model.steps[-1][-1].predict(data.data),
            y_true=data.target.value.to_numpy(),
            label=label,
            encoding=encoding,
            data=data.data,
            meta=data.meta,
            suffix=suffix,
        )
        self.logger.log_params(transfer_object)
        self.logger.log_metrics(transfer_object)
        self.logger.log_artifacts(transfer_object)


class LoggerMixin:
    def save_explanations(self, tobj: TransferObject, explainer_path: Path):
        # if len(tobj.best_model.steps) > 1:
        #     # ? probably not needed anymore
        #     model_temp = deepcopy(tobj.best_model)
        #     model_temp.steps.pop(-1)
        #     data_transformed = model_temp.transform(tobj.data.to_numpy())
        #     column_map = {f"x{idx:03d}": col for idx, col in enumerate(tobj.data.columns)}
        #     data_transformed.columns = [
        #         replace_substring(column_map, col) for col in data_transformed.columns
        #     ]

        # else:
        data_transformed = tobj.data.copy()

        try:
            reg = tobj.best_model.steps[-1][-1]  # decision function (regressor)
            explainer = shap.TreeExplainer(reg)
            joblib.dump(explainer, explainer_path / "explainer.joblib")

            class_names = None
            if hasattr(tobj.best_model, "classes_"):
                class_names = ["".join(tobj.encoding[idx]) for idx in tobj.best_model.classes_]

            # save shap artifacts
            shap_values = explainer.shap_values(data_transformed)
            self._save_shap_figure(shap_values, data_transformed, explainer_path, "bar", class_names)
            self._save_shap_figure(shap_values, data_transformed, explainer_path, "dot", class_names)
            self._save_shap_figure(shap_values, data_transformed, explainer_path, "violin", class_names)
        except Exception as e:
            logging.warning(f"Could not save save shap artifacts: {e}")

        finally:
            plt.close("all")

    def _save_shap_figure(self, shap_values, data, explainer_path, plot_type, class_names=None):
        save_path = explainer_path / f"shap_summary_plot_{plot_type}.pdf"
        with save_plot_figure(save_path):
            shap.summary_plot(shap_values, data, plot_type=plot_type, class_names=class_names)


class ArtifactLoggerClassification(LoggerMixin):
    def log_params(self, tobj: TransferObject):
        mlflow.log_params(tobj.best_trial.params)
        logging.info(f"Hyperparameters used: {tobj.best_trial.params}")

    def log_metrics(self, tobj: TransferObject):
        metrics = calculate_classification_metrics(tobj.y_true, tobj.y_pred)
        mlflow.log_metrics(
            {"".join([tobj.suffix, "_", key]): val for key, val in metrics.to_dict().items()}
        )
        logging.info(f"--> Metrics on {tobj.suffix} data:\n{str(metrics)}")
        logging.info(f"Classification report:\n" f"{classification_report(tobj.y_true, tobj.y_pred)}")

    def log_artifacts(self, tobj: TransferObject):
        with tempfile.TemporaryDirectory(dir=configs.BASE_DIR) as dp:
            model_path = ensure_dir(Path(dp, configs.MLFLOW_MODEL))  # ignore for now  # noqa
            explainer_path = ensure_dir(Path(dp, configs.MLFLOW_EXPLAINER, tobj.suffix))
            results_path = ensure_dir(Path(dp, configs.MLFLOW_RESULTS, tobj.suffix))
            configs_path = ensure_dir(Path(dp, configs.MLFLOW_CONFIGS))

            write_txt(tobj.data.to_string(), Path(dp) / f"data_{tobj.suffix}.txt")
            write_json(tobj.best_trial.params, configs_path / "best_params.json")
            write_txt(
                classification_report(tobj.y_true, tobj.y_pred),
                results_path / "classification_report.txt",
            )
            self._save_confusion_matrix(tobj, results_path)
            # joblib.dump(model_instance, path) # automatically done by mlflow

            self.save_explanations(tobj, explainer_path)
            mlflow.log_artifacts(dp)

    def _save_confusion_matrix(self, tobj: TransferObject, results_path: Path):
        cm = confusion_matrix(tobj.y_true, tobj.y_pred)
        display_labels = ["".join(tobj.encoding[idx]) for idx in tobj.best_model.classes_]
        cm_display = ConfusionMatrixDisplay(cm, display_labels=display_labels)
        cm_display.plot(cmap="Blues", values_format="d")
        plt.savefig(results_path / "confusion_matrix.png")
        plt.close("all")


class ArtifactLoggerRegression(LoggerMixin):
    def log_params(self, tobj: TransferObject):
        mlflow.log_params(tobj.best_trial.params)
        logging.info(f"Hyperparameters used: {tobj.best_trial.params}")

    def log_metrics(self, tobj: TransferObject):
        metrics = calculate_regression_metrics(tobj.y_true, tobj.y_pred)
        mlflow.log_metrics(
            {"".join([tobj.suffix, "_", key]): val for key, val in metrics.to_dict().items()}
        )
        logging.info(f"--> Metrics on {tobj.suffix} data:\n{str(metrics)}")

    def log_artifacts(self, tobj: TransferObject):
        with tempfile.TemporaryDirectory(dir=configs.BASE_DIR) as dp:
            model_path = ensure_dir(Path(dp, configs.MLFLOW_MODEL))  # ignore for now  # noqa
            explainer_path = ensure_dir(Path(dp, configs.MLFLOW_EXPLAINER, tobj.suffix))
            results_path = ensure_dir(Path(dp, configs.MLFLOW_RESULTS, tobj.suffix))
            configs_path = ensure_dir(Path(dp, configs.MLFLOW_CONFIGS))

            write_txt(tobj.data.to_string(), Path(dp) / f"data_{tobj.suffix}.txt")
            write_json(tobj.best_trial.params, configs_path / "best_params.json")
            write_txt(
                f"MSE on {tobj.suffix} data: {np.mean((tobj.y_true - tobj.y_pred) ** 2)}",
                results_path / "mse.txt",
            )
            self.save_explanations(tobj, explainer_path)
            mlflow.log_artifacts(dp)
