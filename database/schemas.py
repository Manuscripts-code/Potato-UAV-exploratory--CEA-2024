from typing import Optional

from sqlmodel import Field, Relationship, SQLModel


class Metric(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str = Field(index=True)
    value: float
    ci_low: Optional[float]  # confidence interval
    ci_high: Optional[float]  # confidence interval

    record: Optional["Record"] = Relationship(back_populates="metrics")
    record_id: Optional[int] = Field(default=None, foreign_key="record.id")


class Artifact(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str = Field(index=True)
    data: bytes

    record: Optional["Record"] = Relationship(back_populates="artifacts")
    record_id: Optional[int] = Field(default=None, foreign_key="record.id")


class Data(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str = Field(index=True)
    data: bytes

    record: Optional["Record"] = Relationship(back_populates="data")
    record_id: Optional[int] = Field(default=None, foreign_key="record.id")


class Record(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    model_name: str = Field(index=True)
    model_version: str

    data: list["Data"] = Relationship(back_populates="record")
    metrics: list["Metric"] = Relationship(back_populates="record")
    artifacts: list["Artifact"] = Relationship(back_populates="record")
