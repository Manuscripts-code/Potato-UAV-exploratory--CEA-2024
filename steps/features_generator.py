import pandas as pd
from rich import print
from sklearn.base import BaseEstimator
from typing_extensions import Annotated
from zenml import step

from data_manager.loaders import StructuredData


@step(enable_cache=False)
def features_generator(
    features_engineer: BaseEstimator,
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

    # indices = data_train.data.index
    data_transformed = features_engineer.transform(data_train.data.to_numpy())
    data_train_feat.data = pd.DataFrame(data=data_transformed)  # , index=indices, columns)

    if data_val is not None and not data_val.data.empty:
        data_transformed = features_engineer.transform(data_val.data.to_numpy())
        data_val_feat.data = pd.DataFrame(data=data_transformed)  # , index=indices, columns=columns)

    if data_test is not None and not data_test.data.empty:
        data_transformed = features_engineer.transform(data_test.data.to_numpy())
        data_test_feat.data = pd.DataFrame(data=data_transformed)  # , index=indices, columns=columns)

    print("Data used for training:\n", data_train_feat.data)
    return data_train_feat, data_val_feat, data_test_feat
