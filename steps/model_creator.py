import logging

from sklearn.pipeline import Pipeline
from typing_extensions import Annotated
from zenml import step
from zenml.client import Client

from configs import configs
from configs.parser import ModelConfig
from models.models import Model


@step(enable_cache=configs.CACHING, experiment_tracker=Client().active_stack.experiment_tracker.name)
def model_creator(model_cfg: ModelConfig) -> Annotated[Pipeline, "model"]:
    logging.info("Creating model...")
    model = Model(model_cfg.pipeline, model_cfg.unions).create()
    logging.info(f"Model created: {model}")
    return model
