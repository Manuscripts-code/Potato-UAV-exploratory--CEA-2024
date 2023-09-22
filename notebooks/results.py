from database.db import SQLiteDatabase
from utils.metrics import calculate_classification_metrics, calculate_regression_metrics
from utils.tools import calculate_metric_and_confidence_interval

db = SQLiteDatabase()

records = db.get_all_records()
records_latest = db.get_latest_records()
pass
