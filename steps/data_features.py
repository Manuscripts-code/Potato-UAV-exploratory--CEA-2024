from typing_extensions import Annotated
from zenml import step

from configs import configs, options
from configs.parser import FeaturesConfig
from data_manager.loaders import StructuredData
from utils.utils import init_object, set_random_seed


@step(enable_cache=False)
def data_features(
    data_train: StructuredData,
    data_val: StructuredData,
    data_test: StructuredData,
    features_cfg: FeaturesConfig,
) -> tuple[
    Annotated[StructuredData, "data_train"],
    Annotated[StructuredData, "data_val"],
    Annotated[StructuredData, "data_test"],
]:
    if features_cfg.features_engineer is None:
        return data_train, data_val, data_test

    features_engineer = init_object(
        options.FEATURE_ENGINEERS, features_cfg.features_engineer, **features_cfg.params()
    )
    set_random_seed(configs.RANDOM_SEED)
    data_train.data = features_engineer.fit_transform(data_train.data, data_train.target.value)
    data_test.data = features_engineer.transform(data_test.data)
    return data_train, data_val, data_test
