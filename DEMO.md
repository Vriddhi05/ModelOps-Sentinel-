# Five-minute interview walkthrough

**0:00–0:40 — Frame the demo.** Open `http://127.0.0.1:8000`. Explain that this is a rebuild using scikit-learn's bundled Iris dataset, with a versioned model artifact, persisted inference audit rows, and feature drift checks. Point out the model version and held-out score displayed below the title.

**0:40–1:30 — Inspect training and serving boundary.** Briefly show `app/train.py`: stratified split with seed 42, pipeline fit on training rows, held-out metrics, and training-only reference statistics. Show `artifacts/model-v1.json`. Explain that `app.main` loads the saved joblib pipeline at startup; requests do not retrain or fit preprocessing.

**1:30–2:20 — Make a prediction.** In the first card click **Load valid example**, then **Run prediction**. Read the predicted species, probabilities, model version, and request UUID. Open the recent predictions table and point out timestamp and latency.

**2:20–3:00 — Show validation.** Change a measurement to `-1` and submit. The API rejects it with a feature-specific supported-range message. Restore a valid value using the example button. Mention exact schema, numeric, finite-value, and range validation.

**3:00–4:10 — Compare drift batches.** Select **Training-reference batch** and analyze; discuss each feature's PSI, standardized mean shift, and thresholds. Then choose **Intentionally shifted** and analyze; point out the flagged features. A flag means the batch distribution differs from the training reference, not that predictive accuracy has degraded.

**4:10–5:00 — Persistence and evidence.** Restart the service while keeping `data/predictions.sqlite3`, reload, and refresh history. Explain that the prediction is returned as successful only after the audit insert commits. To evaluate actual performance change, the operator needs reliable labels for the affected time window and must compare labeled metrics such as accuracy, class-level precision/recall, and calibration.

**Before the interview:** run `python -m app.train`, then `uvicorn app.main:app --reload`; confirm port 8000 is free and create at least one prediction so history is populated.
