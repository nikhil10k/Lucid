import hashlib
import json
import sys

import numpy as np
from PIL import Image


def test_external_records_export(tmp_path, monkeypatch):
    from app.ml import predict
    from app.services import imaging
    from research.trust import export_inputs

    image_path = tmp_path / "external_sample.png"
    Image.new("L", (32, 32), color=128).save(image_path)

    records_path = tmp_path / "external_records.json"
    records = [{"path": str(image_path), "label": -1}]
    records_path.write_text(json.dumps(records), encoding="utf-8")

    split_path = tmp_path / "data" / "processed" / "splits.json"
    split_path.parent.mkdir(parents=True)
    split_data = {"classes": ["NORMAL", "PNEUMONIA"]}
    split_path.write_text(json.dumps(split_data), encoding="utf-8")

    checkpoint_path = tmp_path / "selected_checkpoint.pt"
    checkpoint_path.write_bytes(b"external export test checkpoint")

    captured_paths = []

    def fake_logits(images, checkpoint_path=None):
        captured_paths.append(checkpoint_path)
        return np.tile([[0.1, 0.9]], (len(images), 1))

    def fake_probabilities(images, checkpoint_path=None):
        captured_paths.append(checkpoint_path)
        return np.tile([[0.2, 0.8]], (len(images), 1))

    def fake_features(images, checkpoint_path=None):
        captured_paths.append(checkpoint_path)
        return np.ones((len(images), 512), dtype=float)

    monkeypatch.setattr(predict, "predict_logits", fake_logits)
    monkeypatch.setattr(predict, "predict_proba", fake_probabilities)
    monkeypatch.setattr(predict, "extract_features", fake_features)
    monkeypatch.setattr(
        imaging, "quality", lambda image: {"score": 0.75}
    )
    monkeypatch.chdir(tmp_path)

    output_path = tmp_path / "external_export.npz"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "export_inputs.py",
            "--records", str(records_path),
            "--source", "group_a_external",
            "--out", str(output_path),
            "--checkpoint", str(checkpoint_path),
        ],
    )

    export_inputs.main()

    expected_checkpoint_path = str(checkpoint_path.resolve())
    assert captured_paths == [expected_checkpoint_path] * 3

    expected_checkpoint_hash = hashlib.sha256(
        checkpoint_path.read_bytes()
    ).hexdigest()
    expected_records_hash = hashlib.sha256(
        records_path.read_bytes()
    ).hexdigest()
    expected_splits_hash = hashlib.sha256(
        split_path.read_bytes()
    ).hexdigest()

    with np.load(output_path, allow_pickle=False) as exported:
        assert exported["features"].shape == (1, 512)
        assert exported["logits"].shape == (1, 2)
        assert exported["probs"].shape == (1, 2)
        assert exported["quality"].shape == (1,)
        assert exported["labels"].tolist() == [-1]
        assert exported["source"].item() == "group_a_external"
        assert exported["split"].item() == "external"
        assert exported["checkpoint_sha256"].item() == expected_checkpoint_hash
        assert exported["records_sha256"].item() == expected_records_hash
        assert exported["reference_splits_sha256"].item() == expected_splits_hash
        assert exported["classes"].tolist() == ["NORMAL", "PNEUMONIA"]
