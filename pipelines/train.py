from zenml import pipeline
from zenml.logger import get_logger

from configs import configs
from configs.parser import ConfigParser
from steps import (
    data_formatter,
    data_loader,
    data_sampler,
    model_creator,
    model_evaluator,
    model_optimizer,
    model_register,
)

logger = get_logger(__name__)


@pipeline(enable_cache=configs.CACHING)
def train_and_register_model_pipeline() -> None:
    cfg_parser = ConfigParser()
    logger.info(f"Using toml file: {cfg_parser.toml_cfg_path}")

    data = data_loader(cfg_parser.general(), cfg_parser.multispectral())
    data = data_formatter(data, cfg_parser.general(), cfg_parser.formatter())
    data_train, data_val, data_test = data_sampler(data, cfg_parser.sampler())

    model = model_creator(cfg_parser.model())
    best_model, best_trial = model_optimizer(model, data_train, data_val, cfg_parser.optimizer())
    model_evaluator(best_model, best_trial, data_train, data_val, data_test, cfg_parser.evaluator())

    if configs.REGISTER_MODEL:
        model_register(best_model, cfg_parser.registry())


if __name__ == "__main__":
    train_and_register_model_pipeline()
