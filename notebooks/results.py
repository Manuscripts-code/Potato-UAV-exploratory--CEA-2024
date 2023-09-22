from sklearn.metrics import accuracy_score

from database.db import SQLiteDatabase
from utils.metrics import calculate_classification_metrics, calculate_regression_metrics
from utils.tools import calculate_metric_and_confidence_interval

db = SQLiteDatabase()

records = db.get_all_records()
records_latest = db.get_latest_records()

y_true = records_latest[0].data[1].content.target.value.to_numpy()
y_pred = records_latest[0].predictions[1].content.predictions

results = calculate_metric_and_confidence_interval(y_true, y_pred, accuracy_score)
results = calculate_classification_metrics(y_true, y_pred)
