from datetime import datetime
from typing import Optional

from sqlmodel import Field, Relationship, SQLModel

from configs import configs


class MetricSchema(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str = Field(index=True)
    value: float
    ci_low: Optional[float]  # confidence interval
    ci_high: Optional[float]  # confidence interval

    record: Optional["RecordSchema"] = Relationship(back_populates="metrics")
    record_id: Optional[int] = Field(default=None, foreign_key="recordschema.id")


class PredictionSchema(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str = Field(index=True)
    content: bytes

    record: Optional["RecordSchema"] = Relationship(back_populates="predictions")
    record_id: Optional[int] = Field(default=None, foreign_key="recordschema.id")


class DataSchema(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str = Field(index=True)
    content: bytes

    record: Optional["RecordSchema"] = Relationship(back_populates="data")
    record_id: Optional[int] = Field(default=None, foreign_key="recordschema.id")


class RecordSchema(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    model_name: str = Field(index=True)
    model_version: str
    is_latest: bool = Field(default=True)
    created_at: datetime = Field(default=datetime.now())

    data: list["DataSchema"] = Relationship(back_populates="record")
    metrics: list["MetricSchema"] = Relationship(back_populates="record")
    predictions: list["PredictionSchema"] = Relationship(back_populates="record")
