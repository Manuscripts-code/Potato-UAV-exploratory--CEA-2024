from zenml import pipeline
from zenml.logger import get_logger

from configs import configs
from configs.parser import ConfigParser
from steps import data_formatter, data_loader, data_sampler, service_deployer, service_predictor

logger = get_logger(__name__)


@pipeline(enable_cache=configs.CACHING)
def deployment_inference_pipeline() -> None:
    cfg_parser = ConfigParser()
    logger.info(f"Using toml file: {cfg_parser.toml_cfg_path}")

    data = data_loader(cfg_parser.general(), cfg_parser.multispectral())
    data = data_formatter(data, cfg_parser.formatter())
    data_train, data_val, data_test = data_sampler(data, cfg_parser.sampler())

    model_service = service_deployer(cfg_parser.registry())
    service_predictor(model_service, data_test)


if __name__ == "__main__":
    deployment_inference_pipeline()
