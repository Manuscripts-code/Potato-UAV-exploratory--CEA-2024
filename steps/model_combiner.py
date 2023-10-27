import mlflow
from optuna.trial import FrozenTrial
from rich import print
from sklearn.base import BaseEstimator, clone
from sklearn.discriminant_analysis import StandardScaler
from sklearn.pipeline import Pipeline, make_pipeline
from typing_extensions import Annotated
from xgboost import XGBRegressor
from zenml import step
from zenml.client import Client

from data_manager.features import AutoFeatRegression
from data_manager.loaders import StructuredData
from models.methods import PLSRegressionWrapper
from utils.utils import init_object


@step(enable_cache=False, experiment_tracker=Client().active_stack.experiment_tracker.name)
def model_combiner(
    model: Pipeline,
    features_engineer: BaseEstimator,
    data_train: StructuredData,
    best_trial: FrozenTrial,
) -> Annotated[Pipeline, "best_model"]:
    mlflow.sklearn.autolog()

    model = clone(model)
    model.set_params(**best_trial.params)

    best_model = make_pipeline(features_engineer, *model.named_steps.values())
    best_model.fit(data_train.data.to_numpy(), data_train.target.value.to_numpy())
    return best_model
