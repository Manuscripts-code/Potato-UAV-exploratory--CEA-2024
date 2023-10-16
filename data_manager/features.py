from typing import Literal, Union

import pandas as pd
from autofeat import AutoFeatClassifier, AutoFeatRegressor
from sklearn.base import BaseEstimator, TransformerMixin

from configs import configs
from utils.tools import compute_indices, feature_selector_factory
from utils.utils import set_random_seed


class AutoFeatClassification(AutoFeatClassifier):
    def __init__(self, verbose: int = 0, feateng_steps: int = 1, **kwargs):
        set_random_seed(configs.RANDOM_SEED)
        super().__init__(verbose=verbose, feateng_steps=feateng_steps, n_jobs=1)

    def fit(self, data: pd.DataFrame, target: pd.Series):
        set_random_seed(configs.RANDOM_SEED)
        super().fit(data, target)

    def transform(self, data: pd.DataFrame):
        set_random_seed(configs.RANDOM_SEED)
        data_transformed = super().transform(data)
        data_transformed.index = data.index
        return data_transformed

    def fit_transform(self, data: pd.DataFrame, target: pd.Series):
        set_random_seed(configs.RANDOM_SEED)
        data_transformed = super().fit_transform(data, target)
        data_transformed.index = data.index
        return data_transformed


class AutoFeatRegression(AutoFeatRegressor):
    def __init__(self, verbose: int = 0, feateng_steps: int = 1, **kwargs):
        set_random_seed(configs.RANDOM_SEED)
        super().__init__(verbose=verbose, feateng_steps=feateng_steps, n_jobs=1)

    def fit(self, data: pd.DataFrame, target: pd.Series):
        set_random_seed(configs.RANDOM_SEED)
        super().fit(data, target)

    def transform(self, data: pd.DataFrame):
        set_random_seed(configs.RANDOM_SEED)
        data_transformed = super().transform(data)
        data_transformed.index = data.index
        return data_transformed

    def fit_transform(self, data: pd.DataFrame, target: pd.Series):
        set_random_seed(configs.RANDOM_SEED)
        data_transformed = super().fit_transform(data, target)
        data_transformed.index = data.index
        return data_transformed


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
        selector_generated: Union["AutoFeatClassification", "AutoFeatRegression"],
    ):
        self.selector_spectral_indices = selector_spectral_indices
        self.selector_generated = selector_generated

    def fit(self, data: pd.DataFrame, target: pd.Series) -> BaseEstimator:
        self.selector_generated.fit(data, target)
        self.selector_spectral_indices.fit(data, target)
        return self

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        df_generated = self.selector_generated.transform(data)
        df_spectral_indices = self.selector_spectral_indices.transform(data)
        return pd.concat([df_generated, df_spectral_indices], axis=1)

    def fit_transform(self, data: pd.DataFrame, target: pd.Series) -> pd.DataFrame:
        df_generated = self.selector_generated.fit_transform(data, target)
        df_spectral_indices = self.selector_spectral_indices.fit_transform(data, target)
        return pd.concat([df_generated, df_spectral_indices], axis=1)


class AutoSpectralIndicesPlusGeneratedClassification(AutoSpectralIndicesPlusGenerated):
    def __init__(self, verbose: int = 0, n_jobs=1, feateng_steps: int = 1, **kwargs):
        selector_spectral_indices = AutoSpectralIndicesClassification(
            verbose=verbose, n_jobs=n_jobs, merge_with_original=False
        )
        selector_generated = AutoFeatClassification(verbose=verbose, feateng_steps=feateng_steps)
        super().__init__(
            selector_spectral_indices=selector_spectral_indices,
            selector_generated=selector_generated,
        )


class AutoSpectralIndicesPlusGeneratedRegression(AutoSpectralIndicesPlusGenerated):
    def __init__(self, verbose: int = 0, n_jobs=1, feateng_steps: int = 1, **kwargs):
        selector_spectral_indices = AutoSpectralIndicesRegression(
            verbose=verbose, n_jobs=n_jobs, merge_with_original=False
        )
        selector_generated = AutoFeatRegression(verbose=verbose, feateng_steps=feateng_steps)
        super().__init__(
            selector_spectral_indices=selector_spectral_indices,
            selector_generated=selector_generated,
        )
