from rich import print
from sklearn.base import BaseEstimator
from typing_extensions import Annotated
from zenml import step

from configs import options
from configs.parser import FeaturesConfig
from data_manager.loaders import StructuredData
from models.methods import PLSRegressionWrapper
from utils.utils import init_object


@step(enable_cache=False)
def features_engineer_creator(
    data_train: StructuredData,
    features_cfg: FeaturesConfig = FeaturesConfig(),
) -> Annotated[BaseEstimator, "features_engineer"]:
    # if not features_cfg.features_engineer:
    #     return data_train, data_val, data_test

    # features_engineer: BaseEstimator = init_object(  # TODO: add typehint for feature engineer
    #     options.FEATURE_ENGINEERS, features_cfg.features_engineer, **features_cfg.params()
    # )
    features_engineer = PLSRegressionWrapper().fit(
        data_train.data.to_numpy(), data_train.target.value.to_numpy()
    )
    return features_engineer
