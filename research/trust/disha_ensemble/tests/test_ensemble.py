import numpy as np
from research.trust.disha_ensemble.ensemble import ensemble_stats
from research.trust.disha_ensemble.risk_coverage import risk_coverage


def test_identical_members():
    r = ensemble_stats(np.array([[0.2, 0.2, 0.2], [0.8, 0.8, 0.8]]))
    assert np.allclose(r["std"], 0)
    assert r["member_probs"].shape == (2, 3)


def test_coverage():
    c = np.array([1, 1, 0, 0])
    s = np.array([0.1, 0.2, 0.8, 0.9])
    df = risk_coverage(c, s)
    assert df.iloc[0].accuracy == 0.5
    assert df.iloc[-1].accuracy == 1.0
