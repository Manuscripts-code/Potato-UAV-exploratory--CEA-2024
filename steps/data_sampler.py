from typing_extensions import Annotated
from zenml import step

from configs import options
from configs.parser import SamplerConfig
from data_manager import samplers
from data_manager.loaders import StructuredData
from utils.utils import init_object


@step(enable_cache=False)
def data_sampler(
    data: StructuredData, sampler_cfg: SamplerConfig
) -> tuple[
    Annotated[StructuredData, "data_train"],
    Annotated[StructuredData, "data_val"],
    Annotated[StructuredData, "data_test"],
]:
    splitter = init_object(options.SPLITTERS, sampler_cfg.splitter, **sampler_cfg.params())
    sampler = samplers.Sampler(splitter)
    data_train, data_val, data_test = sampler.sample(data)
    return data_train, data_val, data_test
