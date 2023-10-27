from rich import print
from sklearn.base import BaseEstimator
from sklearn.discriminant_analysis import StandardScaler
from sklearn.pipeline import Pipeline, make_pipeline
from typing_extensions import Annotated
from xgboost import XGBRegressor
from zenml import step

from data_manager.features import AutoFeatRegression
from data_manager.loaders import StructuredData
from models.methods import PLSRegressionWrapper
from utils.utils import init_object


@step(enable_cache=False)
def model_combiner(
    # features_engineer: BaseEstimator,
    best_model: Pipeline,
    data_train: StructuredData,
    data_val: StructuredData,
    data_test: StructuredData,
) -> Pipeline:
    # best_model_ = make_pipeline(features_engineer, *best_model.named_steps.values())
    xgb = XGBRegressor()
    pre = PLSRegressionWrapper()
    best_model_ = Pipeline([("pre", pre), ("xgb", xgb)])
    best_model_.fit(data_train.data.to_numpy(), data_train.target.value.to_numpy())
    return best_model_
