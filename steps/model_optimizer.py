import logging

from optuna.trial import FrozenTrial
from sklearn.pipeline import Pipeline
from typing_extensions import Annotated
from zenml import step
from zenml.client import Client

from configs import configs, options
from configs.parser import OptimizerConfig
from data_structures.schemas import StructuredData
from models.optimizers import Optimizer
from utils.utils import init_object


@step(enable_cache=False, experiment_tracker=Client().active_stack.experiment_tracker.name)
def model_optimizer(
    model: Pipeline, data_train: StructuredData, data_val: StructuredData, optimizer_cfg: OptimizerConfig
) -> tuple[Annotated[Pipeline, "best_model"], Annotated[FrozenTrial, "best_trial"]]:
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
    return optimizer.best_model, optimizer.best_trial
