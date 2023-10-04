import os
import pickle
from typing import Type

import numpy as np
import pandas as pd
from pydantic import BaseModel
from zenml.enums import ArtifactType, VisualizationType
from zenml.io import fileio
from zenml.materializers.base_materializer import BaseMaterializer
from zenml.metadata.metadata_types import DType, MetadataType
from zenml.utils import yaml_utils

from configs import configs


class BaseModel(BaseModel):
    class Config:
        arbitrary_types_allowed = True


class ClassificationTarget(BaseModel):
    label: pd.Series
    value: pd.Series
    encoding: pd.Series

    def __getitem__(self, indices):
        label = self.label.iloc[indices]
        value = self.value.iloc[indices]
        return ClassificationTarget(label=label, value=value, encoding=self.encoding)

    def __len__(self):
        return len(self.label)

    def to_dict(self):
        return {
            "label": self.label.to_list(),
            "value": self.value.to_list(),
            "encoding": self.encoding.to_list(),
        }

    def reset_index(self):
        return ClassificationTarget(
            label=self.label.reset_index(drop=True),
            value=self.value.reset_index(drop=True),
            encoding=self.encoding,
        )

    @classmethod
    def from_dict(cls, data):
        label = pd.Series(data["label"], name="label")
        value = pd.Series(data["value"], name="value")
        encoding = pd.Series(data["encoding"], name="encoding")
        return cls(label=label, value=value, encoding=encoding)


class RegressionTarget(BaseModel):
    value: pd.Series
    name: str

    def __getitem__(self, indices):
        value = self.value.iloc[indices]
        return RegressionTarget(value=value, name=self.name)

    def __len__(self):
        return len(self.value)

    def to_dict(self):
        return {
            "value": self.value.to_list(),
            "name": self.name,
        }

    def reset_index(self):
        return RegressionTarget(value=self.value.reset_index(drop=True), name=self.name)

    @classmethod
    def from_dict(cls, data):
        value = pd.Series(data["value"], name="value")
        name = data["name"]
        return cls(value=value, name=name)


class StructuredData(BaseModel):
    data: pd.DataFrame
    meta: pd.DataFrame
    target: ClassificationTarget | RegressionTarget | None = None

    def __getitem__(self, indices):
        data = self.data.iloc[indices]
        meta = self.meta.iloc[indices]
        target = None if self.target is None else self.target[indices]
        return StructuredData(data=data, meta=meta, target=target)

    def to_dict(self):
        return {
            "data": self.data.to_dict(),
            "meta": self.meta.to_dict(),
            "target": self.target.to_dict() if self.target is not None else None,
        }

    def to_bytes(self):
        return pickle.dumps(self.to_dict())

    def reset_index(self):
        return StructuredData(
            data=self.data.reset_index(drop=True),
            meta=self.meta.reset_index(drop=True),
            target=self.target.reset_index() if self.target is not None else None,
        )

    @classmethod
    def from_dict(cls, data):
        data_ = pd.DataFrame(data["data"])
        meta = pd.DataFrame(data["meta"])
        target = StructuredData.target_from_dict(data["target"])
        return cls(data=data_, meta=meta, target=target)

    @staticmethod
    def target_from_dict(data):
        if data is None:
            return None

        elif "name" in data:
            return RegressionTarget.from_dict(data)

        elif "encoding" in data:
            return ClassificationTarget.from_dict(data)

        else:
            raise ValueError("Invalid target dict.")

    @classmethod
    def from_bytes(cls, data):
        return cls.from_dict(pickle.loads(data))


class StructuredDataMaterializer(BaseMaterializer):
    ASSOCIATED_TYPES = (StructuredData,)
    ASSOCIATED_ARTIFACT_TYPE = ArtifactType.DATA

    def __init__(self, uri: str):
        super().__init__(uri)

    def load(self, data_type: Type[StructuredData]) -> StructuredData:
        data = yaml_utils.read_json(os.path.join(self.uri, configs.MATERIALIZER_DATA_JSON))
        return StructuredData.from_dict(data)

    def save(self, data: StructuredData) -> None:
        yaml_utils.write_json(os.path.join(self.uri, configs.MATERIALIZER_DATA_JSON), data.to_dict())

    def save_visualizations(self, data: StructuredData) -> dict[str, VisualizationType]:
        collected_uris = {}

        #! commented because only one dataframe can be shown obviously
        # describe_data_uri = self._get_pandas_describe_uri(
        #     data.data, configs.MATERIALIZER_DESCRIBE_DATA_CSV
        # )
        # collected_uris[describe_data_uri] = VisualizationType.CSV

        describe_meta_uri = self._get_pandas_describe_uri(
            data.meta, configs.MATERIALIZER_DESCRIBE_META_CSV
        )
        collected_uris[describe_meta_uri] = VisualizationType.CSV

        if data.target is not None:
            describe_target_uri = self._get_pandas_describe_uri(
                pd.DataFrame(data.target.value), configs.MATERIALIZER_DESCRIBE_TARGET_CSV
            )
            collected_uris[describe_target_uri] = VisualizationType.CSV

        return collected_uris

    def _get_pandas_describe_uri(self, df: pd.DataFrame, name_csv: str) -> str:
        describe_uri = os.path.join(self.uri, name_csv)
        with fileio.open(describe_uri, mode="wb") as f:
            df.describe().to_csv(f)
        return describe_uri

    def extract_metadata(self, data: StructuredData) -> dict[str, MetadataType]:
        metadata: dict[str, MetadataType] = {}
        metadata["data_shape"] = data.data.shape
        metadata["data_dtypes"] = data.data.dtypes.apply(lambda x: DType(x.name)).to_dict()
        metadata["meta_dtypes"] = data.meta.dtypes.apply(lambda x: DType(x.name)).to_dict()
        return metadata


class Prediction(BaseModel):
    predictions: np.ndarray
    name: str = ""

    def to_dict(self):
        return {
            "predictions": self.predictions,
            "name": self.name,
        }

    def to_bytes(self):
        return pickle.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data):
        predictions = np.array(data["predictions"])
        name = data["name"]
        return cls(predictions=predictions, name=name)

    @classmethod
    def from_bytes(cls, data):
        return cls.from_dict(pickle.loads(data))
