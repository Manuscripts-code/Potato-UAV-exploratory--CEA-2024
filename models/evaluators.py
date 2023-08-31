import tempfile
from pathlib import Path
from typing import Protocol

import joblib
import mlflow
import mlflow.sklearn

from configs import configs
from utils.utils import ensure_dir, write_json, write_txt


class ArtifactLogger(Protocol):
    def log(self, data: dict):
        ...


class Evaluator:
    def __init__(self, logger: ArtifactLogger):
        self.logger = logger

    def run(self):
        pass


class ArtifactLoggerClassification:
    def log(self):
        # Log metrics and parameters and model
        # mlflow.sklearn.log_model(self.model, "model")
        mlflow.log_params(best_params)
        mlflow.log_metrics({self.scoring_metric: best_metric})

        performance = calculate_classification_metrics(y_test, y_pred)
        mlflow.log_metrics({"precision_avg": performance["overall"]["precision"]})
        mlflow.log_metrics({"recall_avg": performance["overall"]["recall"]})
        mlflow.log_metrics({"f1_avg": performance["overall"]["f1"]})
        mlflow.log_metrics({"accuracy_avg": accuracy_score(y_test, y_pred)})
        clf_report = classification_report(y_test, y_pred)

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

        # log info
        self.logger.info(f"Best hyperparameters found were: {best_params}")
        self.logger.info(f"Best {self.scoring_metric}: {best_metric}")
        self.logger.info(f"Run ID: {self.run_id}")
        self.logger.info(f"Classification report on train data:\n{clf_report}")
