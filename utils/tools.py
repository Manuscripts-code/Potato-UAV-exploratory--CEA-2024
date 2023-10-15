from types import SimpleNamespace
from typing import Callable, Iterable, Literal

import numpy as np
import pandas as pd
import spyndex
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from rich import print
from scipy.stats import bootstrap
from sklearn.linear_model import Ridge, RidgeClassifier
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import RobustScaler

from configs import configs
from configs.constants import SPECTRAL_INDICES


def calculate_confidence_interval(
    data: tuple[np.ndarray, np.ndarray], statistic: Callable[[Iterable, Iterable], float]
):
    try:
        res = bootstrap(
            data,
            statistic=statistic,
            n_resamples=1000,
            confidence_level=0.95,
            random_state=0,
            paired=True,
            vectorized=False,
            method="BCa",
        )
        ci = res.confidence_interval
    except ValueError as e:
        print(e)
        ci = [np.nan, np.nan]
    return ci


def calculate_metric_and_confidence_interval(
    y_true: np.ndarray, y_pred: np.ndarray, metric: Callable[[Iterable, Iterable], float]
):
    """
    Currently not used.
    """
    mean = metric(y_true, y_pred)
    ci = calculate_confidence_interval((y_true, y_pred), statistic=metric)
    return SimpleNamespace(mean=mean, ci=ci)


def compute_indices(data: pd.DataFrame) -> pd.DataFrame:
    """Specific to this particular project and dataset.
    Spectral indices were chosen based on multispectral sensor used (micasense RedEdge-MX)
    """
    df = spyndex.computeIndex(
        index=SPECTRAL_INDICES,
        params={
            configs.BAND_BLUE_S: data[configs.BAND_BLUE],
            configs.BAND_GREEN_S1: data[configs.BAND_GREEN],
            configs.BAND_GREEN_S2: data[configs.BAND_GREEN],
            configs.BAND_RED_S: data[configs.BAND_RED],
            configs.BAND_RED_EDGE_S1: data[configs.BAND_RED_EDGE],
            configs.BAND_RED_EDGE_S2: data[configs.BAND_RED_EDGE],
            configs.BAND_RED_EDGE_S3: data[configs.BAND_RED_EDGE],
            configs.BAND_NIR_S1: data[configs.BAND_NIR],
            configs.BAND_NIR_S2: data[configs.BAND_NIR],
        },
    )
    # Drop columns with inf or NaN values
    df = df.drop(df.columns[df.isin([np.inf, -np.inf, np.nan]).any()], axis=1)
    # Drop columns with constant values
    df = df.drop(df.columns[df.nunique() == 1], axis=1)
    return df


def feature_selector_factory(
    problem_type: Literal["regression", "classification"], verbose: int = 2, n_jobs: int = 1
) -> Pipeline:
    """Specific to this particular project and dataset."""
    if problem_type == "regression":
        algo = Ridge()
        scoring = "neg_mean_squared_error"
    elif problem_type == "classification":
        algo = RidgeClassifier()
        scoring = "f1_weighted"
    else:
        raise ValueError(
            f"Invalid problem type: {problem_type}, possible values are: 'regression' or 'classification'"
        )
    sfs = SFS(
        estimator=algo,
        k_features=(1, 20),  # type: ignore # noqa # can be: "best" - most extensive, (1, n) - check range of features, n - exact number of features
        forward=True,  # selection in forward direction
        floating=True,  # floating algorithm - can go back and remove features once added
        verbose=verbose,
        scoring=scoring,
        cv=3,
        n_jobs=n_jobs,
    )
    return make_pipeline(RobustScaler(), sfs)
