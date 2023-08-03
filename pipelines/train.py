from zenml import pipeline

from configs import configs
from configs.parser import ConfigParser
from steps import data_loader, data_sampler, data_formatter


@pipeline(enable_cache=configs.CACHING)
def train_and_register_model_pipeline() -> None:
    parser = ConfigParser()
    general_config = parser.get_general_configs()
    multispectral_config = parser.get_multispectral_configs()
    sampler_config = parser.get_sampler_configs()
    formatter_config = parser.get_formatter_configs()

    data = data_loader(general_config, multispectral_config)
    data = data_formatter(data, formatter_config)
    data = data_sampler(data, sampler_config)


if __name__ == "__main__":
    train_and_register_model_pipeline()
