from rich import print
from typing_extensions import Annotated
from zenml import step

from data_manager.features import FeaturesEngineer
from data_manager.loaders import StructuredData


@step(enable_cache=False)
def features_generator(
    features_engineer: FeaturesEngineer,
    data_train: StructuredData,
    data_val: StructuredData | None = None,
    data_test: StructuredData | None = None,
) -> tuple[
    Annotated[StructuredData, "data_train_feat"],
    Annotated[StructuredData | None, "data_val_feat"],
    Annotated[StructuredData | None, "data_test_feat"],
]:
    data_train_feat = data_train.copy()
    data_val_feat = data_val.copy() if data_val is not None else None
    data_test_feat = data_test.copy() if data_test is not None else None

    data_train_feat.data = features_engineer.transform(data_train.data)

    if data_val is not None and not data_val.data.empty:
        data_val_feat.data = features_engineer.transform(data_val.data)

    if data_test is not None and not data_test.data.empty:
        data_test_feat.data = features_engineer.transform(data_test.data)

    print("Data used for training:\n", data_train_feat.data)
    return data_train_feat, data_val_feat, data_test_feat
