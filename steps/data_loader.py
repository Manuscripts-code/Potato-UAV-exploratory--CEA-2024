import logging

from zenml import step

from configs import configs
from data_manager.loaders import MultispectralLoader, StructuredData
from configs.parser import GeneralConfig, MultispectralConfig


@step
def data_loader(
    general_cfg: GeneralConfig, multispectral_cfg: MultispectralConfig
) -> StructuredData:
    logging.info("Loading data...")
    loader = MultispectralLoader(
        general_cfg,
        multispectral_cfg,
        save_dir=configs.SAVE_MERGED_DIR,
        save_coords=configs.SAVE_COORDS,
    ).load()
    logging.info("Done")
    return loader.structured_data
