import logging

from sklearn.pipeline import Pipeline
from zenml import step

from configs.parser import ModelConfig
from typing_extensions import Annotated
from models.models import Model





@step()
def model_creator(model_cfg: ModelConfig) -> Pipeline:
    logging.info("Creating model...")
    model = Model(model_cfg.pipeline, model_cfg.unions).create()
    logging.info(f"Model created: {model}")
    return model
