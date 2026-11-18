import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.train import train


@pytest.fixture
def client(tmp_path):
    artifact, _ = train(tmp_path / "artifacts")
    from app.main import create_app
    return TestClient(create_app(artifact, tmp_path / "demo.sqlite3"))


def sample():
    return {"sepal length (cm)": 5.1, "sepal width (cm)": 3.5,
            "petal length (cm)": 1.4, "petal width (cm)": .2}


def test_prediction_and_history(client):
    response = client.post("/api/predict", json={"features": sample()})
    assert response.status_code == 200
    payload = response.json()
    assert payload["prediction"] == 0
    assert payload["model_version"] == "iris-logreg-v1"
    assert payload["prediction_id"] and payload["probabilities"]
    record = client.get("/api/history").json()["items"][0]
    assert record["prediction_id"] == payload["prediction_id"]
    assert record["status"] == "success" and len(record["input_hash"]) == 64


@pytest.mark.parametrize("features", [
    {"sepal length (cm)": 5.1},
    {"sepal length (cm)": True, "sepal width (cm)": 3.5, "petal length (cm)": 1.4, "petal width (cm)": .2},
    {"sepal length (cm)": -1, "sepal width (cm)": 3.5, "petal length (cm)": 1.4, "petal width (cm)": .2},
])
def test_invalid_requests(client, features):
    assert client.post("/api/predict", json={"features": features}).status_code == 422


def test_metadata_and_health(client):
    assert client.get("/api/health").json()["model_loaded"]
    metadata = client.get("/api/metadata").json()
    assert metadata["metadata"]["dataset"] == "scikit-learn Iris"
    assert metadata["evaluation"]["held_out_rows"] == 30
    assert len(metadata["features"]) == 4


def test_persistence_across_app_restart(tmp_path):
    artifact, _ = train(tmp_path / "artifacts")
    db = tmp_path / "persist.sqlite3"
    from app.main import create_app
    first = TestClient(create_app(artifact, db))
    predicted = first.post("/api/predict", json={"features": sample()}).json()
    first.close()
    second = TestClient(create_app(artifact, db))
    assert second.get("/api/history").json()["items"][0]["prediction_id"] == predicted["prediction_id"]


def test_drift_reference_and_shift(client):
    batches = client.get("/api/metadata").json()["sample_batches"]
    reference, shifted = batches["reference_like"], batches["shifted"]
    assert client.post("/api/drift", json={"rows": reference}).json()["drift_detected"] is False
    assert client.post("/api/drift", json={"rows": shifted}).json()["drift_detected"] is True
    assert client.post("/api/drift", json={"rows": [{"bad": 1}]}).status_code == 422
