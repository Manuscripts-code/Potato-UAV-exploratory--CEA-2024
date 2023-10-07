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
    def update_record(
        self,
        model_name: str,
        to_update: dict[str, any],
        model_version: str = None,
        is_latest: bool = None,
    ):
        query = self.session.query(RecordSchema).filter(RecordSchema.model_name == model_name)
        if is_latest is not None:
            query = query.filter(RecordSchema.is_latest == is_latest)
        else:
            query = query.filter(RecordSchema.model_version == model_version)

        if query.count() != 1:
            raise ValueError(f"Expected 1 record, found {query.count()}.")

        record = query.first()
        for key, value in to_update.items():
            setattr(record, key, value)

        self.session.commit()
        self.session.refresh(record)

    @_with_session(no_autoflush=True)
    def get_records(self, model_name: str = None, model_version: str = None, is_latest=None):
        query = self.session.query(RecordSchema)
        if model_name is not None:
            query = query.filter(RecordSchema.model_name == model_name)
        if model_version is not None:
            query = query.filter(RecordSchema.model_version == model_version)
        if is_latest is not None:
            query = query.filter(RecordSchema.is_latest == is_latest)
        records = query.all()
        return self._modify_records(records)
