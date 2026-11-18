
import hashlib
import json
import sys

import numpy as np
import pytest

from app.ml.common import PREPROC_VERSION
from research.trust.niharika_ood.feature_adapter import (
    _validate_checkpoint_metadata,
)

from research.trust.niharika_ood.mahalanobis import (
    fit_mahalanobis,
    mahalanobis_score,
    threshold_at_percentile,
)


def test_shift_and_rank_deficiency():
    rng = np.random.default_rng(42)

    train = rng.normal(size=(30, 8))
    train[:, 7] = train[:, 6]
    labels = np.tile([0, 1], 15)

    params = fit_mahalanobis(train, labels, split="train")
    assert np.isfinite(params["precision"]).all()

    scores = mahalanobis_score(train, params)
    shifted = mahalanobis_score(train + 5.0, params)

    assert scores.shape == (30,)
    assert shifted.shape == (30,)
    assert np.isfinite(scores).all()
    assert np.isfinite(shifted).all()
    assert shifted.mean() > scores.mean()


def test_threshold():
    val_scores = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

    threshold = threshold_at_percentile(val_scores, pct=95)

    assert threshold == pytest.approx(np.percentile(val_scores, 95))


def test_explicit_checkpoint_path_is_used(tmp_path, monkeypatch):
    from research.trust.niharika_ood import feature_adapter

    checkpoint_path = tmp_path / "selected_checkpoint.pt"
    checkpoint_path.write_bytes(b"selected checkpoint")

    class FakeModel:
        def load_state_dict(self, state):
            self.loaded_state = state

        def eval(self):
            return self

    fake_model = FakeModel()

    checkpoint = {
        "state": {"fake": "weights"},
        "arch": "resnet18",
        "classes": ["NORMAL", "PNEUMONIA"],
        "img_size": 224,
        "preprocessing_version": "1.0",
    }

    loaded_paths = []

    def fake_torch_load(path, map_location=None):
        loaded_paths.append(path)
        return checkpoint

    def fake_build_model(arch, n_classes, pretrained=False):
        assert arch == "resnet18"
        assert n_classes == 2
        assert pretrained is False
        return fake_model

    monkeypatch.setattr(
        feature_adapter.torch,
        "load",
        fake_torch_load,
    )
    monkeypatch.setattr(
        feature_adapter,
        "build_model",
        fake_build_model,
    )
    monkeypatch.setattr(feature_adapter, "_MODEL_CACHE", {})

    model, loaded_checkpoint = feature_adapter._get_model(
        checkpoint_path
    )

    assert model is fake_model
    assert loaded_checkpoint is checkpoint
    assert fake_model.loaded_state == checkpoint["state"]
    assert loaded_paths == [checkpoint_path.resolve()]

    # A second request should reuse the cached model.
    feature_adapter._get_model(checkpoint_path)

    assert loaded_paths == [checkpoint_path.resolve()]


def test_exporter_uses_and_hashes_selected_checkpoint(
    tmp_path, monkeypatch
):
    from PIL import Image

    from app.ml import predict
    from app.ml.common import PREPROC_VERSION
    from app.services import imaging
    from research.trust import export_inputs

    data_dir = tmp_path / "data" / "processed"
    data_dir.mkdir(parents=True)

    image_path = tmp_path / "sample.png"
    Image.new("L", (16, 16), color=128).save(image_path)

    splits = {
        "classes": ["NORMAL", "PNEUMONIA"],
        "train": [
            {"path": str(image_path), "label": 0},
        ],
    }

    (data_dir / "splits.json").write_text(
        json.dumps(splits),
        encoding="utf-8",
    )

    checkpoint_path = tmp_path / "selected_checkpoint.pt"
    checkpoint_path.write_bytes(
        b"selected checkpoint for regression test"
    )

    output_path = tmp_path / "exported_features.npz"

    captured_paths = []

    def fake_logits(images, checkpoint_path=None):
        captured_paths.append(checkpoint_path)
        return np.zeros((len(images), 2), dtype=np.float32)

    def fake_probabilities(images, checkpoint_path=None):
        captured_paths.append(checkpoint_path)
        return np.full((len(images), 2), 0.5, dtype=np.float32)

    def fake_features(images, checkpoint_path=None):
        captured_paths.append(checkpoint_path)
        return np.ones((len(images), 512), dtype=np.float32)

    def fake_quality(image):
        return {"score": 0.75}

    monkeypatch.setattr(predict, "predict_logits", fake_logits)
    monkeypatch.setattr(
        predict, "predict_proba", fake_probabilities
    )
    monkeypatch.setattr(
        predict, "extract_features", fake_features
    )
    monkeypatch.setattr(imaging, "quality", fake_quality)

    monkeypatch.chdir(tmp_path)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "export_inputs.py",
            "--split",
            "train",
            "--out",
            str(output_path),
            "--checkpoint",
            str(checkpoint_path),
        ],
    )

    export_inputs.main()

    expected_path = str(checkpoint_path.resolve())

    assert captured_paths == [expected_path] * 3

    expected_hash = hashlib.sha256(
        checkpoint_path.read_bytes()
    ).hexdigest()

    with np.load(output_path, allow_pickle=False) as exported:
        assert str(exported["checkpoint_sha256"]) == expected_hash
        assert exported["features"].shape == (1, 512)
        assert exported["logits"].shape == (1, 2)
        assert exported["probs"].shape == (1, 2)
        assert exported["quality"].shape == (1,)
        assert exported["split"].item() == "train"
        assert exported["classes"].tolist() == [
            "NORMAL",
            "PNEUMONIA",
        ]
        assert (
            exported["preprocessing_version"].item()
            == PREPROC_VERSION
        )



def valid_ood_checkpoint():
    return {
        "state": {},
        "arch": "resnet18",
        "classes": ["NORMAL", "PNEUMONIA"],
        "img_size": 224,
        "preprocessing_version": PREPROC_VERSION,
    }


def test_ood_checkpoint_metadata_is_valid():
    _validate_checkpoint_metadata(valid_ood_checkpoint())


@pytest.mark.parametrize(
    "missing_key",
    [
        "state",
        "arch",
        "classes",
        "img_size",
        "preprocessing_version",
    ],
)
def test_ood_checkpoint_rejects_missing_metadata(missing_key):
    checkpoint = valid_ood_checkpoint()
    del checkpoint[missing_key]

    with pytest.raises(
        ValueError,
        match="missing required metadata",
    ):
        _validate_checkpoint_metadata(checkpoint)


@pytest.mark.parametrize(
    "classes",
    [
        ["PNEUMONIA", "NORMAL"],
        ["NORMAL"],
        ["NORMAL", "OTHER"],
    ],
)
def test_ood_checkpoint_rejects_wrong_class_order(classes):
    checkpoint = valid_ood_checkpoint()
    checkpoint["classes"] = classes

    with pytest.raises(ValueError, match="Expected class order"):
        _validate_checkpoint_metadata(checkpoint)


@pytest.mark.parametrize(
    "version",
    ["0.9", "2.0", None],
)
def test_ood_checkpoint_rejects_preprocessing_mismatch(version):
    checkpoint = valid_ood_checkpoint()
    checkpoint["preprocessing_version"] = version

    with pytest.raises(
        ValueError,
        match="Preprocessing version mismatch",
    ):
        _validate_checkpoint_metadata(checkpoint)
