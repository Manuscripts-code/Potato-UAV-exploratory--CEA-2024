import logging

from zenml import step

from configs import configs
from configs.global_enums import SamplerConfigEnum
from data_manager import samplers
from data_manager.loaders import StructuredData
from configs.parser import SamplerConfig
from utils.utils import init_object

SPLITTERS_OPT = {
    "RandomSplitter": samplers.RandomSplitter,
    "StratifySplitter": samplers.StratifySplitter,
}


@step
def data_sampler(structured_data: StructuredData, sampler_config: SamplerConfig) -> StructuredData:
    splitter_name = sampler_config.splitter
    splitter = init_object(SPLITTERS_OPT, splitter_name)
    sampler = samplers.Sampler(
        splitter,
        split_size_test=sampler_config.split_size_test,
        random_state=sampler_config.random_state,
        shuffle=sampler_config.shuffle,
    )
    return structured_data