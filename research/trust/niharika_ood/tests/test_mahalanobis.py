import numpy as np

from research.trust.niharika_ood.mahalanobis import *
from research.trust.eval_signal import signal_error_auc
from research.trust.niharika_ood import feature_adapter


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


def test_explicit_checkpoint_path_is_used(tmp_path, monkeypatch):
    checkpoint_path = tmp_path / "selected_classifier.pt"
    checkpoint_path.write_bytes(b"test checkpoint")

    expected_checkpoint = {
        "arch": "resnet18",
        "classes": ["NORMAL", "PNEUMONIA"],
        "state": {},
        "img_size": 224,
    }

    class FakeModel:
        def load_state_dict(self, state):
            assert state == {}

        def eval(self):
            return self

    fake_model = FakeModel()
    loaded_paths = []

    def fake_torch_load(path, map_location):
        loaded_paths.append(path)
        assert map_location == "cpu"
        return expected_checkpoint

    def fake_build_model(arch, n_classes, pretrained):
        assert arch == "resnet18"
        assert n_classes == 2
        assert pretrained is False
        return fake_model

    monkeypatch.setattr(feature_adapter.torch, "load", fake_torch_load)
    monkeypatch.setattr(feature_adapter, "build_model", fake_build_model)

    monkeypatch.setattr(feature_adapter, "_MODEL_CACHE", {})

    model, checkpoint = feature_adapter._get_model(checkpoint_path)

    assert model is fake_model
    assert checkpoint is expected_checkpoint
    assert loaded_paths == [checkpoint_path.resolve()]

    # A second request for the same path should reuse the cached model.
    feature_adapter._get_model(checkpoint_path)

    assert loaded_paths == [checkpoint_path.resolve()]


def test_exporter_uses_and_hashes_selected_checkpoint(tmp_path, monkeypatch):
    import hashlib
    import json
    import sys

    import numpy as np
    from PIL import Image

    from research.trust import export_inputs

    # Build a tiny temporary Kermany-style split.
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
    (data_dir / "splits.json").write_text(json.dumps(splits))

    # Use a distinct checkpoint file selected through the CLI.
    checkpoint_path = tmp_path / "selected_checkpoint.pt"
    checkpoint_path.write_bytes(b"selected checkpoint for regression test")

    output_path = tmp_path / "exported_features.npz"

    captured_paths = []

    def fake_extract_batch(images, checkpoint_path=None):
        captured_paths.append(checkpoint_path)
        count = len(images)
        return {
            "features": np.ones((count, 512), dtype=np.float32),
            "logits": np.zeros((count, 2), dtype=np.float32),
            "probs": np.full((count, 2), 0.5, dtype=np.float32),
        }

    monkeypatch.setattr(
        export_inputs,
        "extract_batch",
        fake_extract_batch,
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "export_inputs.py",
            "--split", "train",
            "--out", str(output_path),
            "--checkpoint", str(checkpoint_path),
        ],
    )

    export_inputs.main()

    expected_path = checkpoint_path.resolve()
    assert captured_paths == [expected_path]

    expected_hash = hashlib.sha256(
        checkpoint_path.read_bytes()
    ).hexdigest()

    with np.load(output_path) as exported:
        assert str(exported["checkpoint_sha256"]) == expected_hash
        assert exported["features"].shape == (1, 512)
