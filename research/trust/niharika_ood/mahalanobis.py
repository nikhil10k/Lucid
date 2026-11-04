import numpy as np


def fit_mahalanobis(features, labels, reg=1e-3, split="train"):
    if split != "train":
        raise ValueError("Fit only on training features")
    x = np.asarray(features, float)
    y = np.asarray(labels, int)
    if set(np.unique(y)) != {0, 1}:
        raise ValueError("Need both classes 0 and 1")
    if not np.isfinite(x).all() or reg <= 0:
        raise ValueError("Invalid features/reg")
    means = np.stack([x[y == c].mean(0) for c in (0, 1)])
    centered = x - means[y]
    cov = centered.T @ centered / len(x) + reg * np.eye(x.shape[1])
    return dict(means=means, precision=np.linalg.inv(cov), reg=reg)


def mahalanobis_score(features, params):
    d = np.asarray(features)[:, None, :] - params["means"][None, :, :]
    return np.maximum(
        np.einsum("ncd,de,nce->nc", d, params["precision"], d).min(1), 0
    )


def threshold_at_percentile(val_scores, pct=95):
    return float(np.percentile(val_scores, pct))