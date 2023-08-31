import logging

from sklearn.pipeline import Pipeline
from typing_extensions import Annotated
from zenml import step

from configs import options
from configs.parser import OptimizerConfig
from data_manager.structure import StructuredData
from models.optimizers import Optimizer
from utils.utils import init_object


@step(enable_cache=False)
def model_optimizer(
    model: Pipeline, data_train: StructuredData, data_val: StructuredData, optimizer_cfg: OptimizerConfig
) -> Annotated[Pipeline, "best_model"]:
    logging.info("Optimizing model...")

    validator = init_object(
        options.VALIDATORS,
        optimizer_cfg.validator.validator,
        **optimizer_cfg.validator.params(),
    )

    optimizer = Optimizer(
        data_train=data_train,
        data_val=data_val,
        model=model,
        validator=validator,
        optimizer_cfg=optimizer_cfg,
    )
    optimizer.run()
    return optimizer.model_best
