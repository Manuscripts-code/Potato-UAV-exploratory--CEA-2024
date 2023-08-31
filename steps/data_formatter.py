from typing_extensions import Annotated
from zenml import step

from configs import options
from configs.parser import FormatterConfig
from data_manager import formatter
from data_manager.loaders import StructuredData
from utils.utils import init_object


@step
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
