from zenml import pipeline

from pipelines.steps import data_loader
from utils.config_parser import ConfigParser


@pipeline(enable_cache=True)
def train_and_register_model_pipeline() -> None:
    config_parser = ConfigParser()
    multispectral_config = config_parser.get_multispectral_configs()
    structured_data = data_loader(multispectral_config)
    pass


if __name__ == "__main__":
    train_and_register_model_pipeline()
