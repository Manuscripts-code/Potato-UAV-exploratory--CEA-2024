from sqlmodel import SQLModel, create_engine

from configs import configs
from database import schemas  # noqa: F401 - needs to be imported for SQLModel to create tables

sqlite_url = f"sqlite:///{configs.DB_PATH}"
engine = create_engine(sqlite_url, echo=configs.DB_ECHO)

SQLModel.metadata.create_all(engine)


if __name__ == "__main__":
    import numpy as np
    import pandas as pd
    from sqlmodel import Field, Relationship, Session, SQLModel, create_engine

    from data_structures.schemas import Prediction, StructuredData

    # data with random values
    data = StructuredData(
        data=pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]}),
        meta=pd.DataFrame(
            {"c": [7, 8, 9]},
        ),
    )

    arr = np.array([1, 2, 3])
    prediction = Prediction(predictions=arr, name="test")

    with Session(engine) as session:
        data_table = schemas.Data(name="data", data=data.to_bytes())
        metric_table1 = schemas.Metric(name="accuracy", value=0.9)
        metric_table2 = schemas.Metric(name="f1", value=0.8)
        predictions_table = schemas.Prediction(name="model", predictions=prediction.to_bytes())

        record_table = schemas.Record(
            model_name="test",
            model_version="1.0.0",
            data=[data_table],
            metrics=[metric_table1, metric_table2],
            predictions=[predictions_table],
        )
        session.add(record_table)
        session.commit()
        session.refresh(record_table)
        print("Created record:", record_table)

        data1 = StructuredData.from_bytes(record_table.data[0].data)
        prediction1 = Prediction.from_bytes(record_table.predictions[0].predictions)
        pass
