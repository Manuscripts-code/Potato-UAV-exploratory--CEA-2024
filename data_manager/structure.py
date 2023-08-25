from dataclasses import dataclass

import pandas as pd
from pydantic import BaseModel


class BaseModel(BaseModel):
    class Config:
        arbitrary_types_allowed = True


@dataclass
class Target:
    label: pd.Series
    encoded: pd.Series
    encoding: pd.Series

    def __getitem__(self, indices):
        label = self.label.iloc[indices]
        encoded = self.encoded.iloc[indices]
        return Target(label, encoded, self.encoding)

    def __len__(self):
        return len(self.label)


@dataclass
class StructuredData:
    data: pd.DataFrame
    meta: pd.DataFrame
    target: Target | None = None

    def __getitem__(self, indices):
        data = self.data.iloc[indices]
        meta = self.meta.iloc[indices]
        if self.target is None:
            return StructuredData(data, meta)
        target = self.target[indices]
        return StructuredData(data, meta, target)
