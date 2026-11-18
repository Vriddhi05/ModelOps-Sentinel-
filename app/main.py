"""FastAPI inference service backed by a versioned scikit-learn pipeline."""
import hashlib
import json
import os
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .drift import analyze
from .storage import PredictionStore

ROOT = Path(__file__).resolve().parent.parent


class PredictRequest(BaseModel):
    features: dict[str, Any]


class DriftRequest(BaseModel):
    rows: list[dict[str, Any]]


def load_artifact(path: str | Path):
    path = Path(path)
    if not path.is_file():
        raise RuntimeError(f"Model artifact not found: {path}. Run `python -m app.train` first.")
    try:
        artifact = joblib.load(path)
    except Exception as exc:
        raise RuntimeError(f"Could not load model artifact at {path}: {exc}") from exc
    required = {"pipeline", "metadata", "feature_definitions", "evaluation", "reference_profile", "thresholds", "sample_batches"}
    if not isinstance(artifact, dict) or not required.issubset(artifact):
        raise RuntimeError(f"Model artifact at {path} is incompatible; retrain with `python -m app.train`.")
    return artifact


def create_app(artifact_path: str | Path | None = None, database_path: str | Path | None = None):
    artifact_path = Path(artifact_path or os.getenv("MODEL_ARTIFACT", ROOT / "artifacts" / "model-v1.joblib"))
    database_path = database_path or os.getenv("MODEL_OPS_DB", ROOT / "data" / "predictions.sqlite3")
    artifact = load_artifact(artifact_path)
    store = PredictionStore(database_path)
    store.initialize()
    app = FastAPI(title="ModelOps Sentinel", version=artifact["metadata"]["model_version"])
    app.state.artifact, app.state.store = artifact, store

    @app.get("/api/health")
    def health():
        return {"status": "ok", "model_loaded": True, "model_version": artifact["metadata"]["model_version"]}

    @app.get("/api/metadata")
    def metadata():
        return {"metadata": artifact["metadata"], "features": artifact["feature_definitions"],
                "evaluation": artifact["evaluation"], "sample_batches": artifact["sample_batches"]}

    @app.post("/api/predict")
    def predict(request: PredictRequest):
        definitions = artifact["feature_definitions"]
        names = [x["name"] for x in definitions]
        values = request.features
        if set(values) != set(names):
            missing, extra = sorted(set(names) - set(values)), sorted(set(values) - set(names))
            raise HTTPException(422, detail={"message": "Feature names do not match model schema", "missing": missing, "unexpected": extra})
        ordered = {}
        for definition in definitions:
            name = definition["name"]
            value = values[name]
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise HTTPException(422, detail={"message": f"Feature '{name}' must be a number"})
            if not np.isfinite(value):
                raise HTTPException(422, detail={"message": f"Feature '{name}' must be finite"})
            low, high = definition["supported_range"]
            if value < low or value > high:
                raise HTTPException(422, detail={"message": f"Feature '{name}' must be between {low} and {high}"})
            ordered[name] = float(value)
        start = time.perf_counter()
        row = [[ordered[name] for name in names]]
        prediction = artifact["pipeline"].predict(row)[0]
        probabilities = None
        if hasattr(artifact["pipeline"], "predict_proba"):
            raw = artifact["pipeline"].predict_proba(row)[0]
            classes = artifact["pipeline"].classes_
            probabilities = {str(label): float(prob) for label, prob in zip(classes, raw)}
        latency = (time.perf_counter() - start) * 1000
        pred_value = prediction.item() if hasattr(prediction, "item") else prediction
        pred_id, timestamp = str(uuid.uuid4()), datetime.now(timezone.utc).isoformat()
        canonical = json.dumps(ordered, sort_keys=True, separators=(",", ":"))
        record = {"prediction_id": pred_id, "timestamp": timestamp,
                  "model_version": artifact["metadata"]["model_version"],
                  "input_hash": hashlib.sha256(canonical.encode()).hexdigest(), "latency_ms": latency,
                  "status": "success", "prediction": json.dumps(pred_value),
                  "probabilities": json.dumps(probabilities) if probabilities is not None else None,
                  "features": json.dumps(ordered, sort_keys=True)}
        try:
            store.insert(record)
        except Exception as exc:
            # Do not return a prediction as successful when its required audit record was lost.
            raise HTTPException(503, detail=f"Prediction logging failed; no prediction was returned. Check database availability. ({exc})")
        return {"prediction_id": pred_id, "model_version": record["model_version"],
                "prediction": pred_value, "probabilities": probabilities, "latency_ms": latency}

    @app.get("/api/history")
    def history(limit: int = Query(default=50, ge=1, le=200)):
        return {"items": store.history(limit)}

    @app.post("/api/drift")
    def drift(request: DriftRequest):
        try:
            return analyze(request.rows, artifact["reference_profile"], artifact["thresholds"])
        except ValueError as exc:
            raise HTTPException(422, detail=str(exc)) from exc

    @app.get("/")
    def index():
        return FileResponse(ROOT / "frontend" / "index.html")

    app.mount("/static", StaticFiles(directory=ROOT / "frontend"), name="static")
    return app


app = create_app()
