from zenml import pipeline

from configs import configs
from steps import data_loader, data_sampler
from utils.config_parser import ConfigParser


@pipeline(enable_cache=configs.CACHING)
def train_and_register_model_pipeline() -> None:
    config_parser = ConfigParser()
    general_config = config_parser.get_general_configs()
    multispectral_config = config_parser.get_multispectral_configs()
    sampler_config = config_parser.get_sampler_configs()

    structured_data = data_loader(general_config, multispectral_config)
    structured_data = data_sampler(structured_data, sampler_config)


if __name__ == "__main__":
    train_and_register_model_pipeline()
