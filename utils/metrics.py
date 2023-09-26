from typing import NamedTuple

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)


class ClassificationMetrics(NamedTuple):
    accuracy: float
    precision: float
    recall: float
    f1: float
    confusion_mtx: np.ndarray


class RegressionMetrics(NamedTuple):
    mae: float
    mse: float
    rmse: float
    r2: float


def calculate_classification_metrics(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="weighted")
    recall = recall_score(y_true, y_pred, average="weighted")
    f1 = f1_score(y_true, y_pred, average="weighted")
    confusion_mtx = confusion_matrix(y_true, y_pred)
    return ClassificationMetrics(
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
        confusion_mtx=confusion_mtx,
    )


def calculate_regression_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred, squared=True)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    r2 = r2_score(y_true, y_pred)
    return RegressionMetrics(mae=mae, mse=mse, rmse=rmse, r2=r2)
