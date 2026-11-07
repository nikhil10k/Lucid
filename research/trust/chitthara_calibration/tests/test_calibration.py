
import numpy as np
import pytest

from research.trust.chitthara_calibration.calibration import (
    ece,
    brier,
    nll,
    softmax,
    fit_temperature,
)
from research.trust.chitthara_calibration.run import validate_metadata


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


def make_metadata(split):
    return {
        "split": np.array(split),
        "checkpoint_sha256": np.array("abc123"),
        "splits_sha256": np.array("split123"),
        "classes": np.array(["NORMAL", "PNEUMONIA"]),
        "preprocessing_version": np.array("1.0"),
    }


def test_valid_metadata():
    val = make_metadata("val")
    test = make_metadata("test")

    assert validate_metadata(val, test) is True


def test_missing_metadata():
    val = make_metadata("val")
    test = make_metadata("test")
    del test["splits_sha256"]

    with pytest.raises(ValueError, match="Missing splits_sha256"):
        validate_metadata(val, test)


def test_mismatched_checkpoint():
    val = make_metadata("val")
    test = make_metadata("test")
    test["checkpoint_sha256"] = np.array("different")

    with pytest.raises(ValueError, match="Checkpoint mismatch"):
        validate_metadata(val, test)


def test_mismatched_split_hash():
    val = make_metadata("val")
    test = make_metadata("test")
    test["splits_sha256"] = np.array("different")

    with pytest.raises(ValueError, match="Split file hash mismatch"):
        validate_metadata(val, test)


def test_wrong_split():
    val = make_metadata("test")
    test = make_metadata("test")

    with pytest.raises(ValueError, match="Expected validation split"):
        validate_metadata(val, test)


def test_wrong_class_order():
    val = make_metadata("val")
    test = make_metadata("test")
    test["classes"] = np.array(["PNEUMONIA", "NORMAL"])

    with pytest.raises(ValueError, match="Unexpected class order"):
        validate_metadata(val, test)


def test_wrong_preprocessing_version():
    val = make_metadata("val")
    test = make_metadata("test")
    test["preprocessing_version"] = np.array("2.0")

    with pytest.raises(ValueError, match="Preprocessing version mismatch"):
        validate_metadata(val, test)
