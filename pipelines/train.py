from zenml import pipeline

from configs import configs
from pipelines.steps import data_loader, data_sampler
from utils.config_parser import ConfigParser


@pipeline(enable_cache=configs.CACHING)
def train_and_register_model_pipeline() -> None:
    config_parser = ConfigParser()
    multispectral_config = config_parser.get_multispectral_configs()
    structured_data = data_loader(multispectral_config)
    structured_data = data_sampler(structured_data)


if __name__ == "__main__":
    train_and_register_model_pipeline()
