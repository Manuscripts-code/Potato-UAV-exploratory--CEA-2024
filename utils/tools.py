from types import SimpleNamespace
from typing import Callable, Iterable

import numpy as np
from rich import print
from scipy.stats import bootstrap

"""
Currently not used.
"""


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
    mean = metric(y_true, y_pred)
    ci = calculate_confidence_interval((y_true, y_pred), statistic=metric)
    return SimpleNamespace(mean=mean, ci=ci)
