from typing_extensions import Annotated
from zenml import step

from configs import configs, options
from configs.parser import FormatterConfig, GeneralConfig
from data_manager.loaders import StructuredData
from utils.utils import init_object


@step(enable_cache=configs.CACHING)
def data_formatter(
    data: StructuredData, general_cfg: GeneralConfig, formatter_cfg: FormatterConfig
) -> Annotated[StructuredData, "data"]:
    formatter = init_object(
        options.FORMATTERS,
        formatter_cfg.formatter,
        general_cfg=general_cfg,
        formatter_cfg=formatter_cfg,
    )
    data = formatter.format(data)
    return data
