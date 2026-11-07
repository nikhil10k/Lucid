import numpy as np
import pandas as pd


def risk_coverage(correct, uncertainty, seed=42):
    correct = np.asarray(correct, float)
    s = np.asarray(uncertainty, float)
    order = np.argsort(s, kind="stable")
    rng = np.random.default_rng(seed)
    rows = []
    for c in (1.0, 0.9, 0.8, 0.7, 0.6, 0.5):
        k = max(1, int(round(c * len(correct))))
        random = np.mean(
            [
                correct[rng.permutation(len(correct))[:k]].mean()
                for _ in range(100)
            ]
        )
        rows.append(
            dict(
                coverage=c,
                accuracy=float(correct[order[:k]].mean()),
                accuracy_random=float(random),
            )
        )
    return pd.DataFrame(rows)
