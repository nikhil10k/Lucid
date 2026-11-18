import pytest

from app.ml.common import PREPROC_VERSION
from app.ml.predict import _validate_checkpoint_metadata


def valid_checkpoint():
    return {
        "state": {},
        "arch": "resnet18",
        "classes": ["NORMAL", "PNEUMONIA"],
        "img_size": 224,
        "preprocessing_version": PREPROC_VERSION,
    }


def test_valid_checkpoint_metadata():
    _validate_checkpoint_metadata(valid_checkpoint())


@pytest.mark.parametrize(
    "missing_key",
    ["state", "arch", "classes", "img_size", "preprocessing_version"],
)
def test_missing_metadata_is_rejected(missing_key):
    checkpoint = valid_checkpoint()
    del checkpoint[missing_key]

    with pytest.raises(ValueError, match="missing required metadata"):
        _validate_checkpoint_metadata(checkpoint)


@pytest.mark.parametrize(
    "classes",
    [
        ["PNEUMONIA", "NORMAL"],
        ["NORMAL"],
        ["NORMAL", "OTHER"],
    ],
)
def test_incorrect_class_order_is_rejected(classes):
    checkpoint = valid_checkpoint()
    checkpoint["classes"] = classes

    with pytest.raises(ValueError, match="Expected class order"):
        _validate_checkpoint_metadata(checkpoint)


@pytest.mark.parametrize("version", ["0.9", "2.0", None])
def test_mismatched_preprocessing_version_is_rejected(version):
    checkpoint = valid_checkpoint()
    checkpoint["preprocessing_version"] = version

    with pytest.raises(ValueError, match="Preprocessing version mismatch"):
        _validate_checkpoint_metadata(checkpoint)