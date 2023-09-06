from optuna.trial import FrozenTrial
from sklearn.pipeline import Pipeline
from zenml import step
from zenml.client import Client

from configs import options
from configs.parser import EvaluatorConfig
from data_manager.structure import StructuredData
from models.evaluators import Evaluator
from utils.utils import init_object


@step(enable_cache=False, experiment_tracker=Client().active_stack.experiment_tracker.name)
def model_evaluator(
    best_model: Pipeline,
    best_trial: FrozenTrial,
    data_train: StructuredData,
    data_val: StructuredData,
    data_test: StructuredData,
    evaluator_cfg: EvaluatorConfig,
) -> None:
    logger = init_object(options.LOGGERS, evaluator_cfg.logger)
    evaluator = Evaluator(best_model, best_trial, logger)
    evaluator.run(data_train, "train")
    evaluator.run(data_test, "test")
