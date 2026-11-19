
"""Mahalanobis OOD scoring for the approved Lucid classifier."""

from pathlib import Path

import numpy as np


EXPECTED_CLASSES = ["NORMAL", "PNEUMONIA"]
EXPECTED_PREPROCESSING = "1.0"


def load_ood_artifact(path, expected_metadata):
    """Load OOD statistics and validate their baseline metadata."""
    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(f"OOD artifact not found: {path}")

    with np.load(path, allow_pickle=False) as artifact:
        required = {
            "means",
            "precision",
            "threshold",
            "reg",
            "fit_on",
            "checkpoint_sha256",
            "splits_sha256",
            "classes",
            "preprocessing_version",
        }

        missing = required.difference(artifact.files)
        if missing:
            raise ValueError(f"OOD artifact missing fields: {sorted(missing)}")

        metadata = {
            "checkpoint_sha256": str(artifact["checkpoint_sha256"]),
            "splits_sha256": str(artifact["splits_sha256"]),
            "preprocessing_version": str(artifact["preprocessing_version"]),
        }

        if metadata != expected_metadata:
            raise ValueError("OOD artifact metadata does not match approved baseline")

        classes = artifact["classes"].tolist()
        if classes != EXPECTED_CLASSES:
            raise ValueError(f"Unexpected OOD class order: {classes}")

        if str(artifact["fit_on"]) != "kermany_train":
            raise ValueError("OOD statistics must be fitted on training data")

        means = np.asarray(artifact["means"], dtype=np.float64)
        precision = np.asarray(artifact["precision"], dtype=np.float64)
        threshold = float(artifact["threshold"])
        reg = float(artifact["reg"])

    if means.shape != (2, 512):
        raise ValueError(f"Invalid means shape: {means.shape}")

    if precision.shape != (512, 512):
        raise ValueError(f"Invalid precision shape: {precision.shape}")

    if not np.isfinite(means).all() or not np.isfinite(precision).all():
        raise ValueError("OOD statistics contain non-finite values")

    if not np.isfinite(threshold) or threshold < 0:
        raise ValueError("Invalid OOD threshold")

    if not np.isfinite(reg) or reg <= 0:
        raise ValueError("Invalid OOD regularization")

    return {
        "means": means,
        "precision": precision,
        "threshold": threshold,
        "reg": reg,
    }


def mahalanobis_score(features, params):
    """Return minimum squared Mahalanobis distance across both classes."""
    features = np.asarray(features, dtype=np.float64)

    if features.ndim == 1:
        features = features[None, :]

    if features.ndim != 2 or features.shape[1] != 512:
        raise ValueError("Features must have shape (N, 512)")

    if not np.isfinite(features).all():
        raise ValueError("Features contain non-finite values")

    means = params["means"]
    precision = params["precision"]

    if means.shape != (2, 512) or precision.shape != (512, 512):
        raise ValueError("Invalid OOD parameter dimensions")

    differences = features[:, None, :] - means[None, :, :]

    distances = np.einsum(
        "ncd,de,nce->nc",
        differences,
        precision,
        differences,
    )

    return np.maximum(distances.min(axis=1), 0.0)


def predict_ood(features, params):
    """Return OOD scores and flags; equality with threshold is not OOD."""
    scores = mahalanobis_score(features, params)
    flags = scores > params["threshold"]

    return [
        {
            "ood_score": float(score),
            "ood_flag": bool(flag),
        }
        for score, flag in zip(scores, flags)
    ]
