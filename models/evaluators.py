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
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.pipeline import Pipeline

from configs import configs
from data_structures.schemas import ClassificationTarget, StructuredData
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
        self.best_model = best_model
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
            y_pred=self.best_model.predict(data.data.to_numpy()),
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


class ArtifactLoggerClassification:
    def log_params(self, tobj: TransferObject):
        mlflow.log_params(tobj.best_trial.params)
        logging.info(f"Hyperparameters used: {tobj.best_trial.params}")

    def log_metrics(self, tobj: TransferObject):
        precision, recall, f1, _ = precision_recall_fscore_support(
            tobj.y_true, tobj.y_pred, average="weighted", zero_division=0
        )
        mlflow.log_metrics(
            {
                f"{tobj.suffix}_precision": precision,
                f"{tobj.suffix}_recall": recall,
                f"{tobj.suffix}_f1": f1,
            }
        )
        logging.info(
            f"Classification report on {tobj.suffix} data:\n"
            f"{classification_report(tobj.y_true, tobj.y_pred)}"
        )

    def log_artifacts(self, tobj: TransferObject):
        with tempfile.TemporaryDirectory(dir=configs.BASE_DIR) as dp:
            results_path = ensure_dir(Path(dp, configs.MLFLOW_RESULTS, tobj.suffix))
            configs_path = ensure_dir(Path(dp, configs.MLFLOW_CONFIGS))

            write_json(tobj.best_trial.params, configs_path / "best_params.json")
            write_txt(
                classification_report(tobj.y_true, tobj.y_pred),
                results_path / "classification_report.txt",
            )
            self._save_confusion_matrix(tobj, results_path)
            # joblib.dump(model_instance, path) # automatically done by mlflow

            mlflow.log_artifacts(dp)

    def _save_confusion_matrix(self, tobj: TransferObject, results_path: Path):
        cm = confusion_matrix(tobj.y_true, tobj.y_pred)
        display_labels = ["".join(tobj.encoding[idx]) for idx in tobj.best_model.classes_]
        cm_display = ConfusionMatrixDisplay(cm, display_labels=display_labels)
        cm_display.plot(cmap="Blues", values_format="d")
        plt.savefig(results_path / "confusion_matrix.png")
        plt.close("all")


class ArtifactLoggerRegression:
    def log_params(self, tobj: TransferObject):
        mlflow.log_params(tobj.best_trial.params)
        logging.info(f"Hyperparameters used: {tobj.best_trial.params}")

    def log_metrics(self, tobj: TransferObject):
        mse = np.mean((tobj.y_true - tobj.y_pred) ** 2)
        mlflow.log_metrics({f"{tobj.suffix}_mse": mse})
        logging.info(f"MSE on {tobj.suffix} data: {mse}")

    def log_artifacts(self, tobj: TransferObject):
        with tempfile.TemporaryDirectory(dir=configs.BASE_DIR) as dp:
            model_path = ensure_dir(Path(dp, configs.MLFLOW_MODEL))  # ignore for now  # noqa
            explainer_path = ensure_dir(Path(dp, configs.MLFLOW_EXPLAINER, tobj.suffix))
            results_path = ensure_dir(Path(dp, configs.MLFLOW_RESULTS, tobj.suffix))
            configs_path = ensure_dir(Path(dp, configs.MLFLOW_CONFIGS))

            write_json(tobj.best_trial.params, configs_path / "best_params.json")
            write_txt(
                f"MSE on {tobj.suffix} data: {np.mean((tobj.y_true - tobj.y_pred) ** 2)}",
                results_path / "mse.txt",
            )
            self._save_explanations(tobj, explainer_path)
            mlflow.log_artifacts(dp)

    def _save_explanations(self, tobj: TransferObject, explainer_path: Path):
        if len(tobj.best_model.steps) > 1:
            model_temp = deepcopy(tobj.best_model)
            model_temp.steps.pop(-1)
            data_transformed = model_temp.transform(tobj.data.to_numpy())
            column_map = {f"x{idx:03d}": col for idx, col in enumerate(tobj.data.columns)}
            data_transformed.columns = [
                replace_substring(column_map, col) for col in data_transformed.columns
            ]

        else:
            data_transformed = tobj.data.copy()

        reg = tobj.best_model.steps[-1][-1]  # decision function (regressor)
        explainer = shap.TreeExplainer(reg)
        joblib.dump(explainer, explainer_path / "explainer.joblib")

        # save shap artifacts
        shap_values = explainer.shap_values(data_transformed)
        self._save_shap_figure(shap_values, data_transformed, explainer_path, "dot")
        self._save_shap_figure(shap_values, data_transformed, explainer_path, "bar")
        self._save_shap_figure(shap_values, data_transformed, explainer_path, "violin")

    def _save_shap_figure(self, shap_values, data, explainer_path, plot_type):
        plt.figure()
        shap.summary_plot(shap_values, data, plot_type=plot_type)
        plt.savefig(explainer_path / f"shap_summary_plot_{plot_type}.png")
        plt.close("all")
