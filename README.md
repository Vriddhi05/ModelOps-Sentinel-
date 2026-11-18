# ModelOps Sentinel

ModelOps Sentinel is a **rebuild** of a small AI engineering demo for a live technical interview. It serves a scikit-learn inference pipeline, validates inputs, records auditable predictions in SQLite, and compares incoming feature batches with a training reference profile.

## Dataset and demo purpose

The bundled `sklearn.datasets.load_iris` dataset contains 150 labeled flower measurements. The application predicts one of three Iris species from sepal and petal lengths and widths in centimeters. The dataset and model are for demonstrating tabular inference and monitoring only; this is not a botanical decision system.

## Architecture and behavior

- `app/train.py` makes a reproducible, stratified 80/20 split (`random_state=42`). A `StandardScaler` and `LogisticRegression` pipeline is fit only on the 120 training rows. Evaluation runs on the held-out 30 rows.
- Training writes a versioned joblib pipeline and JSON metadata/profile under `artifacts/`. Serving loads this saved artifact; it does not refit. A missing or incompatible artifact gives an actionable startup error.
- FastAPI provides `/api/health`, `/api/metadata`, `/api/predict`, `/api/history`, and `/api/drift`. The static HTML/JS frontend is served from `/` and calls these same endpoints.
- Each accepted prediction includes a UUID, UTC timestamp, model version, SHA-256 hash of canonicalized input, inference latency, prediction, probabilities, and numeric feature values. SQLite insertion is transactional. If persistence fails, the endpoint returns HTTP 503 and withholds the prediction as a successful response: every returned success has an audit row.
- History lives in `data/predictions.sqlite3` by default. Set `MODEL_OPS_DB` to change the location. The database is generated runtime data and is not tracked by Git.
- Feature schema requires the exact four named numeric inputs, rejects booleans, non-finite values, and values outside the demo range `[0, 20]` cm. Drift requests require exact feature alignment, finite values, and 1–1000 rows.

## Drift method and limits

The training-only reference profile stores per-feature mean, sample standard deviation, and quantile-bin counts. `/api/drift` computes population stability index (PSI) over those reference bins plus the absolute standardized mean shift `abs((batch_mean - training_mean) / training_std)`. Default configurable thresholds in the artifact are PSI `0.20` and mean shift `1.0`; a feature flags if either threshold is met. The response shows both measurements and per-feature thresholds. Two reproducible batches are bundled in the model artifact: the 120 training rows (reference-like) and the same rows with fixed feature offsets (intentionally shifted).

A flag is evidence of a feature distribution change relative to the training split, not proof that prediction quality changed. To assess performance, collect trustworthy labels for the relevant production period and evaluate predictive metrics (for example accuracy, per-class precision/recall, and calibration) against those labels. Population drift alone cannot establish error or causality.

## Install and run

Requires Python 3.12. From the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m app.train
uvicorn app.main:app --reload
```

Open <http://127.0.0.1:8000>. On macOS/Linux, activate with `source .venv/bin/activate`. Dependencies are pinned in `requirements.txt`.

To retrain, run `python -m app.train`. To choose alternate paths, set `MODEL_ARTIFACT` and `MODEL_OPS_DB` environment variables. A model file is bundled so a normal checkout can start immediately; retraining replaces the model artifacts. A deployment should use a persistent volume for the SQLite database.

## Example input

```json
{"features":{"sepal length (cm)":5.1,"sepal width (cm)":3.5,"petal length (cm)":1.4,"petal width (cm)":0.2}}
```

The UI also has a one-click valid example. In the drift panel select either the training-reference batch or intentionally shifted 120-row batch.

## Tests and container

Run focused API, validation, persistence-across-restart, metadata, and drift checks:

```powershell
python -m pytest -q
```

Build and run with Docker:

```powershell
docker build -t modelops-sentinel .
docker run --rm -p 8000:8000 -v modelops-data:/app/data modelops-sentinel
```

## Limitations and troubleshooting

- Iris is a small educational dataset and the metrics do not indicate suitability for real production use.
- SQLite suits this single-process local demo; production workloads need operational backup, concurrency, retention, and access-control policies.
- Logging failure returns 503, so clients should retry carefully and use request IDs or idempotency if integrating a real caller. Prediction IDs are generated per request.
- Drift checks are univariate and do not detect all multivariate changes. Thresholds are demo defaults and require domain calibration.
- If startup reports a missing artifact, run `python -m app.train`. If the UI reports backend unavailable, confirm the uvicorn process and port. If history appears empty after restart, confirm `MODEL_OPS_DB` points at the same persistent file.

See [DEMO.md](DEMO.md) for the exact five-minute walkthrough and [INTERVIEW.md](INTERVIEW.md) for implementation details and trade-offs.
