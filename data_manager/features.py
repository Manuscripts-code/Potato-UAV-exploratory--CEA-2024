from typing import Literal, Union

import numpy as np
import pandas as pd
import spyndex
from autofeat import AutoFeatClassifier, AutoFeatRegressor
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import Ridge, RidgeClassifier
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import RobustScaler

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
        k_features=20,  # can be: "best" - most extensive, [1, n] - check range of features, n - exact number of features # noqa
        forward=True,  # selection in forward direction
        floating=True,  # floating algorithm - can go back and remove features once added
        verbose=verbose,
        scoring=scoring,
        cv=3,
        n_jobs=n_jobs,
    )
    return make_pipeline(RobustScaler(), sfs)


class AutoSpectralIndices(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        problem_type=Literal["regression", "classification"],
        verbose: int = 0,
        n_jobs=1,
        merge_with_original: bool = True,
    ):
        self.merge_with_original = merge_with_original
        self.selector = feature_selector_factory(
            problem_type=problem_type, verbose=verbose, n_jobs=n_jobs
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
        columns_select_idx = list(self.selector[1].k_feature_idx_)
        df_indices = df_indices.iloc[:, columns_select_idx]
        if self.merge_with_original:
            return pd.concat([data, df_indices], axis=1)
        return df_indices

    def fit_transform(self, data: pd.DataFrame, target: pd.Series) -> pd.DataFrame:
        self.fit(data, target)
        return self.transform(data)


class AutoSpectralIndicesClassification(AutoSpectralIndices):
    def __init__(self, verbose: int = 0, n_jobs=1, merge_with_original: bool = True, **kwargs):
        super().__init__(
            problem_type="classification",
            verbose=verbose,
            n_jobs=n_jobs,
            merge_with_original=merge_with_original,
        )


class AutoSpectralIndicesRegression(AutoSpectralIndices):
    def __init__(self, verbose: int = 0, n_jobs=1, merge_with_original: bool = True, **kwargs):
        super().__init__(
            problem_type="regression",
            verbose=verbose,
            n_jobs=n_jobs,
            merge_with_original=merge_with_original,
        )


class AutoSpectralIndicesPlusGenerated(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        selector_spectral_indices: Union[
            "AutoSpectralIndicesPlusGeneratedClassification",
            "AutoSpectralIndicesPlusGeneratedRegression",
        ],
        selector_generated: Union["AutoFeatClassifier", "AutoFeatRegressor"],
    ):
        self.selector_spectral_indices = selector_spectral_indices
        self.selector_generated = selector_generated

    def fit(self, data: pd.DataFrame, target: pd.Series) -> BaseEstimator:
        self.selector_spectral_indices.fit(data, target)
        self.selector_generated.fit(data, target)
        return self

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        df_spectral_indices = self.selector_spectral_indices.transform(data)
        df_generated = self.selector_generated.transform(data)
        df_generated.index = df_spectral_indices.index
        return pd.concat([df_generated, df_spectral_indices], axis=1)

    def fit_transform(self, data: pd.DataFrame, target: pd.Series) -> pd.DataFrame:
        self.fit(data, target)
        return self.transform(data)


class AutoSpectralIndicesPlusGeneratedClassification(AutoSpectralIndicesPlusGenerated):
    def __init__(self, verbose: int = 0, n_jobs=1, feateng_steps: int = 1, **kwargs):
        selector_spectral_indices = AutoSpectralIndicesClassification(
            verbose=verbose, n_jobs=n_jobs, merge_with_original=False
        )
        selector_generated = AutoFeatClassifier(
            verbose=verbose, feateng_steps=feateng_steps, n_jobs=n_jobs
        )
        super().__init__(
            selector_spectral_indices=selector_spectral_indices,
            selector_generated=selector_generated,
        )


class AutoSpectralIndicesPlusGeneratedRegression(AutoSpectralIndicesPlusGenerated):
    def __init__(self, verbose: int = 0, n_jobs=1, feateng_steps: int = 1, **kwargs):
        selector_spectral_indices = AutoSpectralIndicesRegression(
            verbose=verbose, n_jobs=n_jobs, merge_with_original=False
        )
        selector_generated = AutoFeatRegressor(
            verbose=verbose, feateng_steps=feateng_steps, n_jobs=n_jobs
        )
        super().__init__(
            selector_spectral_indices=selector_spectral_indices,
            selector_generated=selector_generated,
        )
