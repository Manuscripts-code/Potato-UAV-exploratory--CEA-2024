from zenml import pipeline
from zenml.logger import get_logger

from configs import configs
from configs.parser import ConfigParser
from steps import (
    data_facets,
    data_formatter,
    data_loader,
    data_sampler,
    db_saver_register,
    features_balancer,
    features_engineer_creator,
    features_generator,
    model_combiner,
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

    data = data_loader(cfg_parser.general().without_varieties(), cfg_parser.multispectral())
    data = data_formatter(data, cfg_parser.general(), cfg_parser.formatter())
    data_train, data_val, data_test = data_sampler(data, cfg_parser.sampler())
    # data_facets(data_train, data_val, data_test)

    features_engineer = features_engineer_creator(data_train, cfg_parser.features())
    data_train_feat, data_val_feat, data_test_feat = features_generator(
        features_engineer, data_train, data_val, data_test
    )
    data_train_feat = features_balancer(data_train_feat, cfg_parser.balancer())

    model = model_creator(cfg_parser.model())
    best_trial = model_optimizer(model, data_train_feat, data_val_feat, cfg_parser.optimizer())
    best_model = model_combiner(model, features_engineer, data_train, best_trial)

    register_step = model_register(best_model, cfg_parser.registry())

    model_evaluator.after(register_step)
    model_evaluator(
        best_model, best_trial, data_train_feat, data_val_feat, data_test_feat, cfg_parser.evaluator()
    )
    db_saver_register.after(register_step)
    db_saver_register(best_trial, cfg_parser.registry())


if __name__ == "__main__":
    train_and_register_model_pipeline()
