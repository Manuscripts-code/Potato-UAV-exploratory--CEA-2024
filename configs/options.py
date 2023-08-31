from sklearn import model_selection
from sklearn.decomposition import PCA
from sklearn.dummy import DummyRegressor
from sklearn.preprocessing import MinMaxScaler, PowerTransformer, StandardScaler
from sklearn.svm import SVC, SVR
from xgboost import XGBClassifier, XGBRegressor

from data_manager import formatter, samplers
from models import loggers, methods

FORMATTERS = {
    "ClassificationFormatter": formatter.ClassificationFormatter,
}
SPLITTERS = {
    "SimpleSplitter": samplers.SimpleSplitter,
}

LOGGERS = {
    "ArtefactLoggerClassification": loggers.ArtifactLoggerClassification,
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
    "PLS": methods.PLSRegressionWrapper,
    "savgol": methods.SavgolWrapper,
    "PCA": PCA,
    "MSC": methods.MSCWrapper,
    "DummyRegressor": DummyRegressor,
    "StandardScaler": StandardScaler,
    "MinMaxScaler": MinMaxScaler,
    "PowerTransformer": PowerTransformer,
}
