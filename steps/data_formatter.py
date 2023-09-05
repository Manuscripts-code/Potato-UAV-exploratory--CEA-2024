from typing_extensions import Annotated
from zenml import step

from configs import configs, options
from configs.parser import FormatterConfig
from data_manager import formatters
from data_manager.loaders import StructuredData
from utils.utils import init_object


@step(enable_cache=configs.CACHING)
def data_formatter(
    data: StructuredData, formatter_cfg: FormatterConfig
) -> Annotated[StructuredData, "data"]:
    formatter_ = init_object(
        options.FORMATTERS,
        formatter_cfg.formatter,
        labels_to_encode=formatter_cfg.labels_to_encode,
    )
    data = formatter_.format(data)
    return data
