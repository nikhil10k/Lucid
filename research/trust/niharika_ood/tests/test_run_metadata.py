import numpy as np
import pytest

from app.ml.common import PREPROC_VERSION
from research.trust.niharika_ood.run import (
    EXPECTED_CLASSES,
    _validate_export_metadata,
)


def valid_export():
    return {
        "classes": np.asarray(EXPECTED_CLASSES),
        "preprocessing_version": np.asarray(PREPROC_VERSION),
    }


def test_valid_export_metadata():
    _validate_export_metadata(valid_export(), "train")


@pytest.mark.parametrize(
    "missing_key",
    ["classes", "preprocessing_version"],
)
def test_export_rejects_missing_metadata(missing_key):
    data = valid_export()
    del data[missing_key]

    with pytest.raises(ValueError, match="missing required metadata"):
        _validate_export_metadata(data, "train")


@pytest.mark.parametrize(
    "classes",
    [
        ["PNEUMONIA", "NORMAL"],
        ["NORMAL"],
        ["NORMAL", "OTHER"],
    ],
)
def test_export_rejects_wrong_class_order(classes):
    data = valid_export()
    data["classes"] = np.asarray(classes)

    with pytest.raises(ValueError, match="incorrect class order"):
        _validate_export_metadata(data, "train")


@pytest.mark.parametrize("version", ["0.9", "2.0", None])
def test_export_rejects_preprocessing_mismatch(version):
    data = valid_export()
    data["preprocessing_version"] = np.asarray(version)

    with pytest.raises(ValueError, match="preprocessing version mismatch"):
        _validate_export_metadata(data, "train")