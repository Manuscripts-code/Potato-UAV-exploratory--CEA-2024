import logging

from typing_extensions import Annotated
from zenml import step

from configs import configs
from configs.parser import GeneralConfig, MultispectralConfig
from data_manager.loaders import MultispectralLoader
from data_manager.structure import StructuredData, StructuredDataMaterializer


@step(enable_cache=configs.CACHING, output_materializers=StructuredDataMaterializer)
def data_loader(
    general_cfg: GeneralConfig, multispectral_cfg: MultispectralConfig
) -> Annotated[StructuredData, "data"]:
    logging.info("Loading data...")
    loader = MultispectralLoader(
        general_cfg,
        multispectral_cfg,
        save_dir=configs.SAVE_MERGED_DIR,
        save_coords=configs.SAVE_COORDS,
        use_reduced_dataset=configs.USE_REDUCED_DATASET,
    ).load()
    return loader.structured_data
