from zenml import step

from configs.parser import FormatterConfig
from data_manager.loaders import MultispectralLoader, StructuredData


@step
def data_formatter(data: StructuredData, formatter_config: FormatterConfig) -> StructuredData:
    return data
