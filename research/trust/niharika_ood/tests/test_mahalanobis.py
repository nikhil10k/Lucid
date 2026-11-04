import numpy as np

from research.trust.niharika_ood.mahalanobis import *
from research.trust.eval_signal import signal_error_auc


def test_shift_and_rank_deficiency():
    rng = np.random.default_rng(42)
    x = rng.normal(size=(200, 8))
    y = np.arange(200) % 2

    x[:, 7] = x[:, 6]

    p = fit_mahalanobis(x, y)

    inside = mahalanobis_score(x, p)
    outside = mahalanobis_score(x + 8, p)

    assert np.isfinite(p["precision"]).all()

    auc = signal_error_auc(
        np.r_[inside, outside],
        np.r_[y * 0, y * 0 + 1],
    )[0]

    assert auc > 0.99


def test_threshold():
    s = np.arange(100)
    t = threshold_at_percentile(s)

    assert (s > t).mean() == 0.05