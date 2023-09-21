from data_structures.schemas import Prediction, StructuredData
from database.db import SQLiteDatabase

db = SQLiteDatabase()

records = db.get_all_records()
pass
