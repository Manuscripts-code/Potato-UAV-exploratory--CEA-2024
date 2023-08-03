import logging

from zenml import step

from configs import configs
from data_loader.loaders import MultispectralLoader, StructuredData
from utils.config_parser import MultispectralConfig


@step
def data_loader(multispectral_config: MultispectralConfig) -> StructuredData:
    logging.info("Loading data...")
    loader = MultispectralLoader(
        multispectral_config,
        save_dir=configs.SAVE_MERGED_DIR,
        save_coords=configs.SAVE_COORDS,
        num_closest_points=configs.NUM_CLOSEST_POINTS,
    ).load()
    logging.info("Done")
    return loader.structured_data
