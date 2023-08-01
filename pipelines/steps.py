from zenml import step

from data_manager.loaders import MultispectralLoader, StructuredData
from utils.config_parser import MultispectralConfig


@step
def data_loader(multispectral_config: MultispectralConfig) -> StructuredData:
    loader = MultispectralLoader(multispectral_config).load()
    return loader.structured_data


if __name__ == "__main__":
    from utils.config_parser import ConfigParser

    config_parser = ConfigParser()
    multispectral_config = config_parser.get_multispectral_configs()
    data_loader(multispectral_config)
