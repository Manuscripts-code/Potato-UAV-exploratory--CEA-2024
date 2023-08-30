import numpy as np
from scipy.signal import savgol_filter
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.dummy import DummyRegressor
from sklearn.preprocessing import MinMaxScaler, PowerTransformer, StandardScaler
from sklearn.svm import SVC, SVR
from xgboost import XGBClassifier, XGBRegressor


class SavgolWrapper(BaseEstimator, TransformerMixin):
    def __init__(self, win_length=7, polyorder=2, deriv=2):
        self.win_length = win_length
        self.polyorder = polyorder
        self.deriv = deriv

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        signatures_sav = []
        sp = [self.win_length, self.polyorder, self.deriv]
        for signal in X:
            if self.win_length != 0:
                signal = savgol_filter(signal, sp[0], sp[1], sp[2])
            signatures_sav.append(signal)
        return np.array(signatures_sav)


class PLSRegressionWrapper(PLSRegression):
    def transform(self, X):
        return super().transform(X)

    def fit_transform(self, X, Y):
        return self.fit(X, Y).transform(X)


class MSCWrapper(BaseEstimator, TransformerMixin):
    """Multiplicative Scatter Correction"""

    def fit(self, X, y=None):
        # mean centr correction
        for i in range(X.shape[0]):
            X[i, :] -= X[i, :].mean()

        # Get the reference spectrum. If not given, estimate it from the mean
        reference = None
        if reference is None:
            # Calculate mean
            self.ref = np.mean(X, axis=0)
        else:
            self.ref = reference
        return self

    def transform(self, X, y=None):
        # Define a new array and populate it with the corrected data
        data_msc = np.zeros_like(X)
        for i in range(X.shape[0]):
            # Run regression
            fit = np.polyfit(self.ref, X[i, :], 1, full=True)
            # Apply correction
            data_msc[i, :] = (X[i, :] - fit[0][1]) / fit[0][0]
        return data_msc


METHODS = {
    "SVC": SVC,
    "SVR": SVR,
    "XGB": XGBClassifier,
    "XGBR": XGBRegressor,
    "PLS": PLSRegressionWrapper,
    "savgol": SavgolWrapper,
    "PCA": PCA,
    "MSC": MSCWrapper,
    "DummyRegressor": DummyRegressor,
    "StandardScaler": StandardScaler,
    "MinMaxScaler": MinMaxScaler,
    "PowerTransformer": PowerTransformer,
}
