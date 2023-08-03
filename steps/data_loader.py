import logging

from zenml import step

from configs import configs
from data_manager.loaders import MultispectralLoader, StructuredData
from utils.config_parser import GeneralConfig, MultispectralConfig


@step
def data_loader(
    general_config: GeneralConfig, multispectral_config: MultispectralConfig
) -> StructuredData:
    logging.info("Loading data...")
    loader = MultispectralLoader(
        general_config,
        multispectral_config,
        save_dir=configs.SAVE_MERGED_DIR,
        save_coords=configs.SAVE_COORDS,
    ).load()
    logging.info("Done")
    return loader.structured_data
