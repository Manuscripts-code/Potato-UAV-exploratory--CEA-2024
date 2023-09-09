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

    @classmethod
    def from_dict(cls, data):
        value = pd.Series(data["value"], name="value")
        name = data["name"]
        return cls(value=value, name=name)


class Group(BaseModel):
    label = pd.Series
    encoded = pd.Series
    encoding = pd.Series

    def __getitem__(self, indices):
        label = self.label.iloc[indices]
        encoded = self.encoded.iloc[indices]
        return Group(label=label, encoded=encoded, encoding=self.encoding)

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
    target: ClassificationTarget | RegressionTarget | None = None
    group: Group | None = None

    def __getitem__(self, indices):
        data = self.data.iloc[indices]
        meta = self.meta.iloc[indices]
        target = None if self.target is None else self.target[indices]
        group = None if self.group is None else self.group[indices]
        return StructuredData(data=data, meta=meta, target=target, group=group)

    def to_dict(self):
        return {
            "data": self.data.to_dict(),
            "meta": self.meta.to_dict(),
            "target": self.target.to_dict() if self.target is not None else None,
            "group": self.group.to_dict() if self.group is not None else None,
        }

    @classmethod
    def from_dict(cls, data):
        data_ = pd.DataFrame(data["data"])
        meta = pd.DataFrame(data["meta"])
        target = StructuredData.target_from_dict(data["target"])
        group = None if data["group"] is None else Group.from_dict(data["group"])
        return cls(data=data_, meta=meta, target=target, group=group)

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


class StructuredDataMaterializer(BaseMaterializer):
    ASSOCIATED_TYPES = (StructuredData,)
    ASSOCIATED_ARTIFACT_TYPE = ArtifactType.DATA

    def load(self, data_type: Type[StructuredData]) -> StructuredData:
        data = yaml_utils.read_json(os.path.join(self.uri, "structured_data.json"))
        return StructuredData.from_dict(data)

    def save(self, my_obj: StructuredData) -> None:
        yaml_utils.write_json(os.path.join(self.uri, "structured_data.json"), my_obj.to_dict())
