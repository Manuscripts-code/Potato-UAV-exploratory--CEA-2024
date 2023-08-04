from zenml import step

from configs.parser import FormatterConfig
from data_manager import formatter
from data_manager.loaders import StructuredData
from utils.utils import init_object

FORMATTER_OPT = {
    "ClassificationFormatter": formatter.ClassificationFormatter,
}


@step
def data_formatter(data: StructuredData, formatter_cfg: FormatterConfig) -> StructuredData:
    formatter_name = formatter_cfg.formatter
    formatter_ = init_object(
        FORMATTER_OPT,
        formatter_name,
        labels_to_encode=formatter_cfg.Labels_to_encode,
    )
    data = formatter_.format(data)
    return data
