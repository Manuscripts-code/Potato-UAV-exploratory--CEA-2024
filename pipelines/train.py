from zenml import pipeline

from configs import configs
from configs.parser import ConfigParser
from steps import data_formatter, data_loader, data_sampler


@pipeline(enable_cache=configs.CACHING)
def train_and_register_model_pipeline() -> None:
    parser = ConfigParser()
    general_config = parser.general()
    multispectral_config = parser.multispectral()
    sampler_config = parser.sampler()
    formatter_config = parser.formatter()

    data = data_loader(general_config, multispectral_config)
    data = data_formatter(data, formatter_config)
    data = data_sampler(data, sampler_config)


if __name__ == "__main__":
    train_and_register_model_pipeline()
