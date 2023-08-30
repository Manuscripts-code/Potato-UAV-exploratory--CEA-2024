import logging

from sklearn.pipeline import Pipeline
from typing_extensions import Annotated
from zenml import step

from configs.parser import OptimizerConfig
from data_manager.structure import StructuredData


@step
def model_optimizer(
    model: Pipeline, data_train: StructuredData, data_val: StructuredData, optimizer_cfg: OptimizerConfig
) -> Annotated[Pipeline, "best_model"]:
    logging.info("Optimizing model...")
    model.fit(data_train.X, data_train.y)
    logging.info("Done")
    return model
