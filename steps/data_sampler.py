import logging

from zenml import step

from configs import configs
from data_manager.loaders import StructuredData


@step
def data_sampler(structured_data: StructuredData) -> StructuredData:
    return structured_data
