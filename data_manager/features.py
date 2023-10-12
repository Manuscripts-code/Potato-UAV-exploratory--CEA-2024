from typing import Literal

import numpy as np
import pandas as pd
import spyndex
from autofeat import AutoFeatClassifier, AutoFeatRegressor
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import Ridge, RidgeClassifier, LogisticRegression, LinearRegression

from configs import configs
from configs.constants import SPECTRAL_INDICES


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
) -> SFS:
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
    return SFS(
        estimator=algo,
        k_features=20,  # can be: "best" - most extensive, [1, n] - check range of features, n - exact number of features # noqa
        forward=True,  # selection in forward direction
        floating=True,  # floating algorithm - can go back and remove features once added
        verbose=verbose,
        scoring=scoring,
        cv=5,
        n_jobs=n_jobs,
    )


class AutoSpectralIndicesClassification(BaseEstimator, TransformerMixin):
    def __init__(self, verbose: int = 0, n_jobs=1, merge_with_original: bool = True, **kwargs):
        self.merge_with_original = merge_with_original
        self.selector = feature_selector_factory(
            problem_type="classification", verbose=verbose, n_jobs=n_jobs
        )

    def fit(self, data: pd.DataFrame, target: pd.Series) -> BaseEstimator:
        df_indices = compute_indices(data)
        self.selector.fit(df_indices, target)
        return self
        """ -- Check plot: performance vs number of features --
        import matplotlib.pyplot as plt
        from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs
        plot_sfs(self.selector.get_metric_dict(), kind='std_err', figsize=(30, 20))
        plt.savefig('selection.png')
        plt.close()
        """

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        df_indices = compute_indices(data)
        features_names = list(self.selector.k_feature_names_)
        df_indices = df_indices[features_names]
        if self.merge_with_original:
            return pd.concat([data, df_indices], axis=1)
        return df_indices

    def fit_transform(self, data: pd.DataFrame, target: pd.Series) -> pd.DataFrame:
        self.fit(data, target)
        return self.transform(data)
