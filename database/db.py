from sqlmodel import Session, SQLModel, create_engine

from configs import configs
from database import schemas  # noqa: F401 - needs to be imported for SQLModel to create tables


class SQLiteDatabase:
    def __init__(self):
        sqlite_url = f"sqlite:///{configs.DB_PATH}"
        self.engine = create_engine(sqlite_url, echo=configs.DB_ECHO)
        SQLModel.metadata.create_all(self.engine)

    def save_record(self, record):
        with Session(self.engine) as session:
            session.add(record)
            session.commit()
            session.refresh(record)

    def get_record(self, model_name, model_version):
        with Session(self.engine) as session:
            record = (
                session.query(schemas.RecordSchema)
                .filter(schemas.RecordSchema.model_name == model_name)
                .filter(schemas.RecordSchema.model_version == model_version)
                .first()
            )
            return record

    def get_records(self, model_name):
        with Session(self.engine) as session:
            records = (
                session.query(schemas.RecordSchema)
                .filter(schemas.RecordSchema.model_name == model_name)
                .all()
            )
            return records

    def get_all_records(self):
        with Session(self.engine) as session:
            records = session.query(schemas.RecordSchema).all()
            return records


if __name__ == "__main__":
    import numpy as np
    import pandas as pd
    from sqlmodel import Field, Relationship, Session, SQLModel, create_engine

    from data_structures.schemas import Prediction, StructuredData
    from database.db import SQLiteDatabase

    # data with random values
    data = StructuredData(
        data=pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]}),
        meta=pd.DataFrame(
            {"c": [7, 8, 9]},
        ),
    )

    arr = np.array([1, 2, 3])
    prediction = Prediction(predictions=arr, name="test")

    data_table = schemas.DataSchema(name="data", data=data.to_bytes())
    metric_table1 = schemas.MetricSchema(name="accuracy", value=0.9)
    metric_table2 = schemas.MetricSchema(name="f1", value=0.8)
    predictions_table = schemas.PredictionSchema(name="model", predictions=prediction.to_bytes())

    record_table = schemas.RecordSchema(
        model_name="test",
        model_version="1.0.0",
        data=[data_table],
        # metrics=[metric_table1, metric_table2],
        predictions=[predictions_table],
    )

    db = SQLiteDatabase()
    db.save_record(record_table)
