from sqlmodel import SQLModel, create_engine

from configs import configs
from database import schemas  # noqa: F401 - needs to be imported for SQLModel to create tables

sqlite_url = f"sqlite:///{configs.DB_PATH}"
engine = create_engine(sqlite_url, echo=configs.DB_ECHO)

SQLModel.metadata.create_all(engine)


if __name__ == "__main__":
    from sqlmodel import Field, Relationship, Session, SQLModel, create_engine

    with Session(engine) as session:
        model_table = schemas.Model(name="test", version="1.0.0")
        metric_table1 = schemas.Metric(name="accuracy", value=0.9)
        metric_table2 = schemas.Metric(name="f1", value=0.8)
        artifact_table1 = schemas.Artifact(name="model", data=b"model")
        artifact_table2 = schemas.Artifact(name="data", data=b"data")

        record_table = schemas.Record(
            models=[model_table],
            metrics=[metric_table1, metric_table2],
            artifacts=[artifact_table1, artifact_table2],
        )
        session.add(record_table)
        session.commit()
        session.refresh(record_table)
        print("Created record:", record_table)
