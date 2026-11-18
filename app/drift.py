"""Reference-profile population stability and standardized mean shift checks."""
from typing import Any
import numpy as np


def analyze(rows: list[dict[str, float]], profile: dict[str, Any], thresholds: dict[str, float]):
    names = profile["feature_names"]
    if not rows:
        raise ValueError("At least one row is required")
    if len(rows) > 1000:
        raise ValueError("Batch size must be 1000 rows or fewer")
    for i, row in enumerate(rows):
        if set(row) != set(names):
            raise ValueError(f"Row {i + 1} must contain exactly these features: {', '.join(names)}")
        for name, value in row.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
                raise ValueError(f"Row {i + 1} feature '{name}' must be a finite number")
            bounds = profile["features"][name]["supported_range"]
            if value < bounds[0] or value > bounds[1]:
                raise ValueError(f"Row {i + 1} feature '{name}' must be between {bounds[0]} and {bounds[1]}")
    result = []
    for name in names:
        ref = profile["features"][name]
        values = np.array([r[name] for r in rows], dtype=float)
        edges = np.array(ref["bin_edges"], dtype=float)
        ref_counts = np.array(ref["bin_counts"], dtype=float)
        cur_counts, _ = np.histogram(values, bins=edges)
        # Values outside the reference histogram are added to edge bins.
        cur_counts[0] += int(np.sum(values < edges[0]))
        cur_counts[-1] += int(np.sum(values > edges[-1]))
        eps = 1e-6
        p = np.maximum(ref_counts / ref_counts.sum(), eps)
        q = np.maximum(cur_counts / cur_counts.sum(), eps)
        psi = float(np.sum((q - p) * np.log(q / p)))
        scale = max(float(ref["std"]), 1e-9)
        mean_shift = float((values.mean() - ref["mean"]) / scale)
        psi_limit, shift_limit = thresholds["psi"], thresholds["standardized_mean_shift"]
        result.append({"feature": name, "batch_mean": float(values.mean()), "reference_mean": ref["mean"],
                       "standardized_mean_shift": mean_shift, "psi": psi,
                       "thresholds": {"psi": psi_limit, "absolute_standardized_mean_shift": shift_limit},
                       "flagged": bool(psi >= psi_limit or abs(mean_shift) >= shift_limit)})
    return {"batch_size": len(rows), "method": "PSI on training-reference quantile bins plus absolute standardized mean shift",
            "thresholds": thresholds, "drift_detected": any(x["flagged"] for x in result), "features": result,
            "interpretation": "A flag indicates a feature distribution differs from the training reference; it does not establish model accuracy degradation."}
