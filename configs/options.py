from autofeat import AutoFeatClassifier, AutoFeatRegressor
from sklearn import model_selection
from sklearn.decomposition import PCA
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler, PowerTransformer, StandardScaler
from sklearn.svm import SVC, SVR
from xgboost import XGBClassifier, XGBRegressor

from data_manager import formatters, samplers
from models import evaluators, methods

FORMATTERS = {
    "ClassificationFormatter": formatters.ClassificationFormatter,
    "RegressionFormatter": formatters.RegressionFormatter,
}
SPLITTERS = {
    "SimpleSplitter": samplers.SimpleSplitter,
}

LOGGERS = {
    "ArtifactLoggerClassification": evaluators.ArtifactLoggerClassification,
    "ArtifactLoggerRegression": evaluators.ArtifactLoggerRegression,
}

VALIDATORS = {
    "RepeatedStratifiedKFold": model_selection.RepeatedStratifiedKFold,
    "RepeatedKFold": model_selection.RepeatedKFold,
}

METHODS = {
    "SVC": SVC,
    "SVR": SVR,
    "XGBClassifier": XGBClassifier,
    "XGBRegressor": XGBRegressor,
    "RandomForestClassifier": RandomForestClassifier,
    "RandomForestRegressor": RandomForestRegressor,
    "PLS": methods.PLSRegressionWrapper,
    "savgol": methods.SavgolWrapper,
    "PCA": PCA,
    "MSC": methods.MSCWrapper,
    "DummyRegressor": DummyRegressor,
    "DummyClassifier": DummyClassifier,
    "StandardScaler": StandardScaler,
    "MinMaxScaler": MinMaxScaler,
    "PowerTransformer": PowerTransformer,
    "AutoFeatClassifier": AutoFeatClassifier,
    "AutoFeatRegressor": AutoFeatRegressor,
}
