# Engineering notes for the interview

## Request flow

The browser fetches model metadata to build the labeled form from the loaded artifact's feature definitions. On submit, it sends JSON to `POST /api/predict`. FastAPI checks the exact feature set, rejects booleans and non-numeric or non-finite values, enforces supported ranges, and orders values according to the artifact's schema. The persisted scikit-learn pipeline predicts and, when available, calculates class probabilities. The API returns the prediction, UUID, model version, and latency only after SQLite commits its audit row.

## Preprocessing and artifact loading

Training builds one scikit-learn `Pipeline(StandardScaler, LogisticRegression)`. `train_test_split` is stratified with a fixed seed. Calling `fit` only on `X_train` fits the scaler's means and scales on training data; pipeline prediction applies those same learned transformations. Held-out metrics are computed without fitting on held-out examples. Joblib stores the fitted pipeline alongside model version, training data reference, feature schema and ranges, held-out metrics, per-feature training profile, and drift thresholds. Startup refuses to serve if the artifact is absent or has an incompatible shape. The artifact is trusted local build output: loading pickle-based joblib from untrusted sources is unsafe.

## Schema validation

The API requires exactly the four feature names. Values must be JSON numbers (not booleans), finite, and between 0 and 20 cm. Batch drift validation applies the same alignment, numeric, finite, and range requirements to every row and limits input to 1000 rows. The UI uses labels and units supplied by metadata and renders server-side errors inline.

## SQLite transactions and failure behavior

Each store operation owns a short-lived SQLite connection. Python's connection context manager commits the insert on success and rolls it back on exceptions, then closes the connection. Prediction audit rows include UUID, UTC timestamp, model version, canonical-input SHA-256, latency, status, output, optional probabilities, and feature values. A successful result is never sent if the insert fails; the endpoint returns HTTP 503. This chooses audit completeness over returning an unlogged result and should be paired with client retry/idempotency policy in a production system. The file path is configurable so an operator can retain the database across restarts.

## Drift calculations

Training statistics are calculated from `X_train` only. Quantile edges define a histogram reference per feature. PSI is `sum((q - p) * ln(q / p))` over reference and current proportions, with a small floor to avoid zero-proportion logs. Current values beyond the edge range are included in the nearest edge bin. The other statistic is signed standardized mean shift `(batch_mean - reference_mean) / reference_std`. A feature flags when PSI is at least 0.20 **or** absolute shift is at least 1.0. Both values and limits are returned. The endpoint is descriptive monitoring: univariate feature changes do not directly test prediction error or causality. Labeled outcome data is needed to examine performance.

## Trade-offs and limitations

- Iris plus LogisticRegression keeps the example credential-free, fast, explainable, and portable; it is not a realistic production target.
- SQLite and one local process keep setup small. A multi-replica service needs shared durable storage and a carefully designed logging path.
- Univariate PSI and mean shifts are easy to explain but can miss interactions and depend on sample size and threshold choices.
- The static frontend avoids a separate JavaScript build chain and communicates only with the real API.
- Model version is a stable demo label (`iris-logreg-v1`); production training should derive versions from immutable run IDs and artifact digests.
