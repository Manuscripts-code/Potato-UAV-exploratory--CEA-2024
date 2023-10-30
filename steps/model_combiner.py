import logging

import mlflow
from optuna.trial import FrozenTrial
from sklearn.base import clone
from sklearn.pipeline import Pipeline, make_pipeline
from typing_extensions import Annotated
from zenml import step
from zenml.client import Client

from data_manager.features import FeaturesEngineer
from data_manager.loaders import StructuredData


@step(enable_cache=False, experiment_tracker=Client().active_stack.experiment_tracker.name)
def model_combiner(
    model: Pipeline,
    features_engineer: FeaturesEngineer,
    data_train: StructuredData,
    best_trial: FrozenTrial,
) -> Annotated[Pipeline, "best_model"]:
    mlflow.sklearn.autolog()

    model = clone(model)
    model.set_params(**best_trial.params)

    best_model = make_pipeline(features_engineer, *model.named_steps.values())
    best_model.fit(data_train.data, data_train.target.value)
    logging.info(f"Features generated:\n {best_model.steps[0][1].transform(data_train.data).columns}")
    logging.info(f"Model used for prediction (as a service): {best_model}")
    return best_model
