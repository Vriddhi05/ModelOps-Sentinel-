"""Train and persist the deterministic Iris demo pipeline and reference profile."""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def train(output_dir: str | Path = Path(__file__).resolve().parent.parent / "artifacts"):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data = load_iris()
    X, y = data.data, data.target
    feature_names = list(data.feature_names)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42, stratify=y)
    # Pipeline fit occurs only on training rows; transform at evaluation/inference reuses this fitted scaler.
    pipeline = Pipeline([("scale", StandardScaler()), ("classifier", LogisticRegression(max_iter=500, random_state=42))])
    pipeline.fit(X_train, y_train)
    predicted = pipeline.predict(X_test)
    evaluation = {"held_out_rows": int(len(y_test)), "accuracy": float(accuracy_score(y_test, predicted)),
                  "classification_report": classification_report(y_test, predicted, target_names=list(data.target_names), output_dict=True)}
    features = []
    profile = {"feature_names": feature_names, "rows": int(len(X_train)), "features": {}}
    for index, name in enumerate(feature_names):
        train_values = X_train[:, index].astype(float)
        edges = np.quantile(train_values, np.linspace(0, 1, 11))
        # Histogram requires strictly increasing edges even for repeated quantiles.
        edges = np.unique(edges)
        if len(edges) < 2:
            edges = np.array([train_values.min() - .5, train_values.max() + .5])
        counts, edges = np.histogram(train_values, bins=edges)
        low, high = 0.0, 20.0
        features.append({"name": name, "label": name.replace(" (cm)", "").replace(" ", " ").title(),
                         "unit": "cm", "supported_range": [low, high]})
        profile["features"][name] = {"mean": float(train_values.mean()), "std": float(train_values.std(ddof=1)),
                                     "min": float(train_values.min()), "max": float(train_values.max()),
                                     "bin_edges": edges.tolist(), "bin_counts": counts.tolist(),
                                     "supported_range": [low, high]}
    reference_batch = [{name: float(value) for name, value in zip(feature_names, row)} for row in X_train]
    offsets = {feature_names[0]: 1.3, feature_names[1]: .55, feature_names[2]: 1.2, feature_names[3]: .8}
    shifted_batch = [{name: float(row[name] + offsets[name]) for name in feature_names} for row in reference_batch]
    artifact = {"pipeline": pipeline,
        "metadata": {"model_version": "iris-logreg-v1", "created_at": datetime.now(timezone.utc).isoformat(),
                     "dataset": "scikit-learn Iris", "purpose": "Demonstrate tabular multiclass flower-species inference; not intended for real-world botanical decisions.",
                     "estimator": "StandardScaler + LogisticRegression", "split": {"test_size": .2, "random_state": 42, "stratified": True},
        "training_rows": int(len(X_train)), "feature_names": feature_names,
                     "class_names": list(data.target_names), "training_data_reference": "Bundled sklearn.datasets.load_iris; profile calculated from training split only."},
        "feature_definitions": features, "evaluation": evaluation, "reference_profile": profile,
        "sample_batches": {"reference_like": reference_batch, "shifted": shifted_batch},
        "thresholds": {"psi": .2, "standardized_mean_shift": 1.0}}
    output = output_dir / "model-v1.joblib"
    joblib.dump(artifact, output)
    summary = {key: val for key, val in artifact.items() if key != "pipeline"}
    (output_dir / "model-v1.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return output, evaluation


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=str(Path(__file__).resolve().parent.parent / "artifacts"))
    args = parser.parse_args()
    path, metrics = train(args.output_dir)
    print(f"Saved model artifact: {path}")
    print(f"Held-out accuracy: {metrics['accuracy']:.3f} ({metrics['held_out_rows']} rows)")
