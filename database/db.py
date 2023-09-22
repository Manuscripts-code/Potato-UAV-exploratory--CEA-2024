from sqlmodel import Session, SQLModel, create_engine

from configs import configs
from data_structures.schemas import Prediction, StructuredData
from database import schemas  # noqa: F401 - needs to be imported for SQLModel to create tables
from database.schemas import RecordSchema


class SQLiteDatabase:
    def __init__(self):
        sqlite_url = f"sqlite:///{configs.DB_PATH}"
        self.engine = create_engine(sqlite_url, echo=configs.DB_ECHO)
        SQLModel.metadata.create_all(self.engine)

    def save_record(self, record: RecordSchema):
        with Session(self.engine) as session:
            session.add(record)
            session.commit()
            session.refresh(record)

    def get_record(self, model_name: str, model_version: str):
        with Session(self.engine) as session:
            record = (
                session.query(RecordSchema)
                .filter(RecordSchema.model_name == model_name)
                .filter(RecordSchema.model_version == model_version)
                .first()
            )
            return record

    def reset_latest_record(self, model_name: str):
        with Session(self.engine) as session:
            (
                session.query(RecordSchema)
                .filter(RecordSchema.model_name == model_name)
                .filter(RecordSchema.is_latest == True)  # noqa: E712
                .update({"is_latest": False})
            )
            session.commit()

    def get_all_records(self):
        with Session(self.engine).no_autoflush as session:
            records = session.query(RecordSchema).all()
            records = self._modify_records(records)
            return records

    def get_latest_records(self):
        with Session(self.engine).no_autoflush as session:
            records = (
                session.query(RecordSchema).filter(RecordSchema.is_latest == True).all()  # noqa: E712
            )
            records = self._modify_records(records)
            return records

    def _modify_records(self, records: list[RecordSchema]):
        for record in records:
            for data in record.data:
                data.content = StructuredData.from_bytes(data.content)
            for prediction in record.predictions:
                prediction.content = Prediction.from_bytes(prediction.content)
        return records
