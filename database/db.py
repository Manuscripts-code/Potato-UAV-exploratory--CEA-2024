from functools import wraps

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
        self.session = None

    def _with_session(no_autoflush=False):
        def decorator(func):
            @wraps(func)
            def wrapper(self, *args, **kwargs):
                with Session(self.engine) as session:
                    self.session = session
                    if no_autoflush:
                        self.session.autoflush = False
                    return func(self, *args, **kwargs)

            return wrapper

        return decorator

    def _modify_records(self, records: list[RecordSchema]):
        for record in records:
            for data in record.data:
                data.content = StructuredData.from_bytes(data.content)
            for prediction in record.predictions:
                prediction.content = Prediction.from_bytes(prediction.content)
        return records

    @_with_session()
    def save_record(self, record: RecordSchema):
        self.session.add(record)
        self.session.commit()
        self.session.refresh(record)

    @_with_session()
    def get_record(self, model_name: str, model_version: str):
        record = (
            self.session.query(RecordSchema)
            .filter(RecordSchema.model_name == model_name)
            .filter(RecordSchema.model_version == model_version)
            .first()
        )
        return record

    @_with_session()
    def reset_latest_record(self, model_name: str):
        (
            self.session.query(RecordSchema)
            .filter(RecordSchema.model_name == model_name)
            .filter(RecordSchema.is_latest == True)  # noqa: E712
            .update({"is_latest": False})
        )
        self.session.commit()

    @_with_session(no_autoflush=True)
    def get_all_records(self):
        records = self.session.query(RecordSchema).all()
        records = self._modify_records(records)
        return records

    @_with_session(no_autoflush=True)
    def get_latest_records(self):
        records = (
            self.session.query(RecordSchema).filter(RecordSchema.is_latest == True).all()  # noqa: E712
        )
        records = self._modify_records(records)
        return records
