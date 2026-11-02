import numpy as np
from sklearn.metrics import roc_auc_score


def signal_error_auc(signal, wrong, n_boot=500, seed=42):
    s = np.asarray(signal, float)
    y = np.asarray(wrong, int)

    if s.ndim != 1 or y.shape != s.shape or not len(y):
        raise ValueError("Need equal, nonempty 1D signal and wrong arrays")

    if not np.isfinite(s).all() or not np.isin(y, [0, 1]).all():
        raise ValueError("Signals must be finite; wrong must be 0 or 1")

    if len(np.unique(y)) < 2:
        return None, None, None  # AUC undefined: all correct or all wrong

    if n_boot < 1:
        raise ValueError("n_boot must be positive")

    rng = np.random.default_rng(seed)
    vals = []

    for _ in range(n_boot):
        ids = rng.integers(0, len(y), len(y))
        if len(np.unique(y[ids])) == 2:
            vals.append(roc_auc_score(y[ids], s[ids]))

    if not vals:
        return float(roc_auc_score(y, s)), None, None

    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(roc_auc_score(y, s)), float(lo), float(hi)