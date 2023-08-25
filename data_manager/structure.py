import os
from typing import Type

import pandas as pd
from pydantic import BaseModel
from zenml.enums import ArtifactType
from zenml.materializers.base_materializer import BaseMaterializer
from zenml.utils import yaml_utils


class BaseModel(BaseModel):
    class Config:
        arbitrary_types_allowed = True


class Target(BaseModel):
    label: pd.Series
    encoded: pd.Series
    encoding: pd.Series

    def __getitem__(self, indices):
        label = self.label.iloc[indices]
        encoded = self.encoded.iloc[indices]
        return Target(label=label, encoded=encoded, encoding=self.encoding)

    def __len__(self):
        return len(self.label)

    def to_dict(self):
        return {
            "label": self.label.to_list(),
            "encoded": self.encoded.to_list(),
            "encoding": self.encoding.to_list(),
        }

    @classmethod
    def from_dict(cls, data):
        label = pd.Series(data["label"], name="label")
        encoded = pd.Series(data["encoded"], name="encoded")
        encoding = pd.Series(data["encoding"], name="encoding")
        return cls(label=label, encoded=encoded, encoding=encoding)


class StructuredData(BaseModel):
    data: pd.DataFrame
    meta: pd.DataFrame
    target: Target | None = None

    def __getitem__(self, indices):
        data = self.data.iloc[indices]
        meta = self.meta.iloc[indices]
        if self.target is None:
            return StructuredData(data=data, meta=meta)
        target = self.target[indices]
        return StructuredData(data=data, meta=meta, target=target)

    def to_dict(self):
        return {
            "data": self.data.to_dict(),
            "meta": self.meta.to_dict(),
            "target": self.target.to_dict() if self.target is not None else None,
        }

    @classmethod
    def from_dict(cls, data):
        data_ = pd.DataFrame(data["data"])
        meta = pd.DataFrame(data["meta"])
        if data["target"] is None:
            return cls(data=data_, meta=meta, target=None)
        target = Target.from_dict(data["target"])
        return cls(data=data_, meta=meta, target=target)


class StructuredDataMaterializer(BaseMaterializer):
    ASSOCIATED_TYPES = (StructuredData,)
    ASSOCIATED_ARTIFACT_TYPE = ArtifactType.DATA

    def load(self, data_type: Type[StructuredData]) -> StructuredData:
        data = yaml_utils.read_json(os.path.join(self.uri, "structured_data.json"))
        return StructuredData.from_dict(data)

    def save(self, my_obj: StructuredData) -> None:
        yaml_utils.write_json(os.path.join(self.uri, "structured_data.json"), my_obj.to_dict())
