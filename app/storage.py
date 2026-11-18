"""Small SQLite persistence layer for prediction audit records."""
import json
import sqlite3
from pathlib import Path
from typing import Any


class PredictionStore:
    def __init__(self, path: str | Path):
        self.path = str(path)

    def initialize(self) -> None:
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as db:
            db.execute("""CREATE TABLE IF NOT EXISTS predictions (
                prediction_id TEXT PRIMARY KEY, timestamp TEXT NOT NULL,
                model_version TEXT NOT NULL, input_hash TEXT NOT NULL,
                latency_ms REAL NOT NULL, status TEXT NOT NULL,
                prediction TEXT NOT NULL, probabilities TEXT,
                features TEXT NOT NULL)""")

    def _connect(self):
        db = sqlite3.connect(self.path, timeout=5)
        db.row_factory = sqlite3.Row
        return db

    def insert(self, record: dict[str, Any]) -> None:
        with self._connect() as db:
            db.execute("""INSERT INTO predictions VALUES
                (:prediction_id,:timestamp,:model_version,:input_hash,:latency_ms,
                 :status,:prediction,:probabilities,:features)""", record)

    def history(self, limit: int = 50) -> list[dict[str, Any]]:
        with self._connect() as db:
            rows = db.execute("SELECT * FROM predictions ORDER BY timestamp DESC LIMIT ?", (limit,)).fetchall()
        result = []
        for row in rows:
            item = dict(row)
            item["prediction"] = json.loads(item["prediction"])
            item["probabilities"] = json.loads(item["probabilities"]) if item["probabilities"] else None
            item["features"] = json.loads(item["features"])
            result.append(item)
        return result
