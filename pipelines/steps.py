from zenml import step

from configs import configs
from data_manager.loaders import MultispectralLoader, StructuredData
from utils.config_parser import MultispectralConfig


@step
def data_loader(multispectral_config: MultispectralConfig) -> StructuredData:
    loader = MultispectralLoader(
        multispectral_config,
        save_dir=configs.SAVE_MERGED_DIR,
        save_coords=configs.SAVE_COORDS,
        num_closest_points=configs.NUM_CLOSEST_POINTS,
    ).load()
    return loader.structured_data


@step
def data_sampler(structured_data: StructuredData) -> StructuredData:
    return structured_data


if __name__ == "__main__":
    from utils.config_parser import ConfigParser

    config_parser = ConfigParser()
    multispectral_config = config_parser.get_multispectral_configs()
    data_loader(multispectral_config)
