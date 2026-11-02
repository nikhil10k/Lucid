import numpy as np
import pytest

from research.trust.chitthara_calibration.calibration import (
    ece,
    brier,
    nll,
    softmax,
    fit_temperature,
)


def test_known_metrics():
    p = np.array([[0.5, 0.5], [0.5, 0.5]])
    y = np.array([0, 1])

    assert ece(p, y) == 0
    assert brier(p, y) == 0.5
    assert np.isclose(nll(np.zeros((2, 2)), y), np.log(2))


def test_temperature_and_class():
    rng = np.random.default_rng(42)
    z = rng.normal(size=(12000, 2)) * 3
    y = (rng.random(len(z)) < softmax(z, 2)[:, 1]).astype(int)
    t = fit_temperature(z, y)

    assert abs(t - 2) < 0.3
    assert np.array_equal(softmax(z, t).argmax(1), z.argmax(1))

    with pytest.raises(ValueError):
        fit_temperature(z, y, split="test")