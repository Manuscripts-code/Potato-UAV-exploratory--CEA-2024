import logging

from typing_extensions import Annotated
from zenml import step

from configs import options
from configs.parser import FeaturesConfig
from data_manager.features import DummyFeaturesGenerator, FeaturesEngineer
from data_manager.loaders import StructuredData
from utils.utils import init_object


@step(enable_cache=False)
def features_engineer_creator(
    data_train: StructuredData,
    features_cfg: FeaturesConfig = FeaturesConfig(),
) -> Annotated[FeaturesEngineer, "features_engineer"]:
    if not features_cfg.features_engineer:
        features_engineer_internal = DummyFeaturesGenerator()
    else:
        features_engineer_internal = init_object(
            options.FEATURE_ENGINEERS, features_cfg.features_engineer, **features_cfg.params()
        )
    features_engineer = FeaturesEngineer(features_engineer_internal)
    features_engineer.fit(data_train.data, data_train.target.value)
    logging.info(f"Feature engineer used: {features_engineer}")
    return features_engineer
