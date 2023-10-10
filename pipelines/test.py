from zenml import pipeline
from zenml.logger import get_logger

from configs import configs
from configs.parser import ConfigParser
from steps import (
    data_facets,
    data_features,
    data_formatter,
    data_loader,
    data_sampler,
    db_saver_deployer,
    service_deployer,
    service_predictor,
)

logger = get_logger(__name__)


@pipeline(enable_cache=configs.CACHING)
def deployment_inference_pipeline() -> None:
    cfg_parser = ConfigParser()
    logger.info(f"Using toml file: {cfg_parser.toml_cfg_path}")

    data = data_loader(cfg_parser.general().without_varieties(), cfg_parser.multispectral())
    data = data_formatter(data, cfg_parser.general(), cfg_parser.formatter())
    data_train, data_val, data_test = data_sampler(data, cfg_parser.sampler())
    data_train, data_val, data_test = data_features(data_train, data_val, data_test, cfg_parser.features())  # type: ignore # noqa
    data_facets(data_train, data_val, data_test)

    model_service = service_deployer(cfg_parser.registry())
    predictions_train = service_predictor(model_service, data_train, cfg_parser.registry())
    predictions_test = service_predictor(model_service, data_test, cfg_parser.registry())
    db_saver_deployer(model_service, data_train, data_test, predictions_train, predictions_test)


if __name__ == "__main__":
    deployment_inference_pipeline()
