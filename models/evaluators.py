import logging
import tempfile
from pathlib import Path
from typing import Protocol

import joblib
import mlflow
import mlflow.sklearn
from optuna.trial import FrozenTrial
from sklearn.metrics import classification_report, precision_recall_fscore_support
from sklearn.pipeline import Pipeline

from configs import configs
from data_manager.structure import StructuredData
from utils.utils import ensure_dir, write_json, write_txt


class ArtifactLogger(Protocol):
    def log_metrics():
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

    def run(self, data: StructuredData, suffix: str = ""):
        y_pred = self.best_model.predict(data.data)
        y_true = data.target.encoded
        label = data.target.label
        encoding = data.target.encoding
        meta = data.meta
        self.logger.log_metrics(self.best_trial, y_pred, y_true, label, encoding, meta, suffix)


class ArtifactLoggerClassification:
    def log_metrics(self, best_trial, y_pred, y_true, label, encoding, meta, suffix):
        mlflow.log_params(best_trial.params)

        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="weighted", zero_division=0
        )
        mlflow.log_metrics(
            dict(
                precision=precision,
                recall=recall,
                f1=f1,
            )
        )
        clf_report = classification_report(y_true, y_pred)

        logging.info(f"Hyperparameters used: {best_trial.params}")
        logging.info(f"Classification report on {suffix} data:\n{clf_report}")

    def log_artifacts(self):
        # Log artifacts
        with tempfile.TemporaryDirectory(dir=configs.BASE_DIR) as dp:
            ensure_dir(Path(dp) / "results")
            ensure_dir(Path(dp) / "configs")
            ensure_dir(Path(dp) / "study")
            ensure_dir(Path(dp) / "model")

            write_json(best_params, Path(dp, "configs/best_params.json"))
            write_json(study_best_params, Path(dp, "study/study_best_params.json"))
            write_json({self.scoring_metric: best_metric}, Path(dp, "results/best_valid_metric.json"))
            write_json(performance, Path(dp, "results/performance.json"))
            write_json(self.config, Path(dp, "configs/config.json"))
            write_txt(study_df, Path(dp, "study/study_df.txt"))
            write_txt(clf_report, Path(dp, "results/classification_report.txt"))
            joblib.dump(self.label_encoder, Path(dp, "model/label_encoder.pkl"))
            joblib.dump(self.data, Path(dp, "model/data.pkl"))
            joblib.dump(self.model, Path(dp, "model/model.pkl"))

            mlflow.log_artifacts(dp)
