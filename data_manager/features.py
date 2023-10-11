import numpy as np
import pandas as pd
import spyndex
from autofeat import AutoFeatClassifier, AutoFeatRegressor, FeatureSelector
from sklearn.base import BaseEstimator, TransformerMixin

from configs import configs
from configs.constants import SPECTRAL_INDICES
from utils.utils import init_object, set_random_seed


def compute_indices(data: pd.DataFrame) -> pd.DataFrame:
    """Specific to this particular project and dataset."""
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


class AutoSpectralIndicesClassification(BaseEstimator, TransformerMixin):
    def __init__(self, feateng_steps: int = 1, verbose: int = 0, merge_with_original: bool = True):
        self.feateeng_steps = feateng_steps
        self.verbose = verbose
        self.merge_with_original = merge_with_original
        self.selector = FeatureSelector(
            verbose=self.verbose, problem_type="classification", featsel_runs=5
        )

    def fit(self, data: pd.DataFrame, target: pd.Series) -> BaseEstimator:
        df_indices = compute_indices(data)
        self.selector.fit(df_indices, target)
        return self

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        df_indices = compute_indices(data)
        df_indices = self.selector.transform(df_indices)
        df_indices.index = data.index
        if self.merge_with_original:
            return pd.concat([data, df_indices], axis=1)
        return df_indices

    def fit_transform(self, data: pd.DataFrame, target: pd.Series) -> pd.DataFrame:
        df_indices = compute_indices(data)
        df_indices = self.selector.fit_transform(df_indices, target)
        df_indices.index = data.index
        if self.merge_with_original:
            return pd.concat([data, df_indices], axis=1)
        return df_indices
