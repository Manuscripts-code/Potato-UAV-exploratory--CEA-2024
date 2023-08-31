import logging

from typing_extensions import Annotated
from zenml import step

from configs import configs
from configs.parser import GeneralConfig, MultispectralConfig
from data_manager.loaders import MultispectralLoader
from data_manager.structure import StructuredData, StructuredDataMaterializer


@step(output_materializers=StructuredDataMaterializer)
def data_loader(
    general_cfg: GeneralConfig, multispectral_cfg: MultispectralConfig
) -> Annotated[StructuredData, "data"]:
    logging.info("Loading data...")
    loader = MultispectralLoader(
        general_cfg,
        multispectral_cfg,
        save_dir=configs.SAVE_MERGED_DIR,
        save_coords=configs.SAVE_COORDS,
    ).load()
    return loader.structured_data
