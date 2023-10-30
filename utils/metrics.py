from typing import NamedTuple

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    max_error,
    mean_absolute_error,
    mean_absolute_percentage_error,
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

    def __str__(self) -> str:
        return (
            f"Accuracy: {self.accuracy:.2f}\n"
            f"Precision: {self.precision:.2f}\n"
            f"Recall: {self.recall:.2f}\n"
            f"F1: {self.f1:.2f}\n"
        )

    def to_dict(self) -> dict:
        return self._asdict()


class RegressionMetrics(NamedTuple):
    mae: float
    mse: float
    rmse: float
    r2: float
    mape: float
    maxe: float

    def __str__(self) -> str:
        return (
            f"Mean absolute error: {self.mae:.2f}\n"
            f"Mean squared error: {self.mse:.2f}\n"
            f"Root mean squared error: {self.rmse:.2f}\n"
            f"R2 score: {self.r2:.2f}\n"
            f"Mean absolute percentage error: {self.mape:.2f}\n"
            f"Max error: {self.maxe:.2f}"
        )

    def to_dict(self) -> dict:
        return self._asdict()


def calculate_classification_metrics(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="weighted")
    recall = recall_score(y_true, y_pred, average="weighted")
    f1 = f1_score(y_true, y_pred, average="weighted")
    return ClassificationMetrics(
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
    )


def calculate_regression_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred, squared=True)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    r2 = r2_score(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    maxe = max_error(y_true, y_pred)
    return RegressionMetrics(
        mae=mae,
        mse=mse,
        rmse=rmse,
        r2=r2,
        mape=mape,
        maxe=maxe,
    )
