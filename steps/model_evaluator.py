import logging

from optuna.trial import FrozenTrial
from sklearn.pipeline import Pipeline
from typing_extensions import Annotated
from zenml import step

from configs import options
from configs.parser import EvaluatorConfig
from data_manager.structure import StructuredData
from models.evaluators import Evaluator
from utils.utils import init_object


@step(enable_cache=False)
def model_evaluator(
    best_model: Pipeline,
    best_trial: FrozenTrial,
    data_train: StructuredData,
    data_val: StructuredData,
    data_test: StructuredData,
    evaluator_cfg: EvaluatorConfig,
) -> None:
    logger = init_object(options.LOGGERS, evaluator_cfg.logger)
    evaluator = Evaluator(logger)
    evaluator.run()
