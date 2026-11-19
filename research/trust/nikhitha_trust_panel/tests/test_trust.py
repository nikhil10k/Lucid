
import io

import numpy as np
import pytest
from PIL import Image
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.routers.trust import router
from research.trust.nikhitha_trust_panel.verdict import decide


CFG = {
    "quality_low": 40,
    "quality_caution": 60,
    "uncertainty_caution": 0.15,
}

ALL_ARTIFACTS = {
    "temperature": True,
    "ensemble": True,
    "ood": True,
}

VALID_SIGNALS = {
    "quality_score": 80,
    "calibrated_confidence": 0.95,
    "uncertainty": 0.02,
    "ood_score": 0.25,
    "ood_flag": False,
}


def test_verdicts():
    assert decide(VALID_SIGNALS, ALL_ARTIFACTS, CFG)[0] == "ok"

    low_quality = dict(VALID_SIGNALS, quality_score=50)
    assert decide(low_quality, ALL_ARTIFACTS, CFG)[0] == "caution"

    ood_detected = dict(VALID_SIGNALS, ood_flag=True)
    assert decide(ood_detected, ALL_ARTIFACTS, CFG)[0] == "low_reliability"

    missing_artifact = dict(ALL_ARTIFACTS, ood=False)
    assert decide(VALID_SIGNALS, missing_artifact, CFG)[0] == "unavailable"


@pytest.mark.parametrize(
    "signal",
    [
        "quality_score",
        "calibrated_confidence",
        "uncertainty",
        "ood_score",
        "ood_flag",
    ],
)
def test_missing_signal_returns_unavailable(signal):
    signals = dict(VALID_SIGNALS)
    signals[signal] = None

    verdict, label, reasons = decide(
        signals, ALL_ARTIFACTS, CFG
    )

    assert verdict == "unavailable"
    assert label == "Unavailable: trust artifacts not found"
    assert any(signal in reason for reason in reasons)


@pytest.mark.parametrize(
    "signal,value",
    [
        ("quality_score", 101),
        ("quality_score", -1),
        ("calibrated_confidence", 1.5),
        ("calibrated_confidence", -0.1),
        ("uncertainty", float("nan")),
        ("ood_score", float("inf")),
        ("ood_flag", 1),
    ],
)
def test_invalid_signal_returns_unavailable(signal, value):
    signals = dict(VALID_SIGNALS)
    signals[signal] = value

    assert decide(signals, ALL_ARTIFACTS, CFG)[0] == "unavailable"


def test_missing_threshold_returns_unavailable():
    config = dict(CFG)
    del config["quality_low"]

    verdict, _, reasons = decide(
        VALID_SIGNALS, ALL_ARTIFACTS, config
    )

    assert verdict == "unavailable"
    assert any("quality_low" in reason for reason in reasons)


def test_invalid_threshold_returns_unavailable():
    config = dict(CFG, quality_low=80, quality_caution=40)

    assert decide(VALID_SIGNALS, ALL_ARTIFACTS, config)[0] == "unavailable"


def test_invalid_config_type_returns_unavailable():
    assert decide(VALID_SIGNALS, ALL_ARTIFACTS, None)[0] == "unavailable"


def test_missing_artifact_returns_unavailable():
    artifacts = dict(ALL_ARTIFACTS)
    artifacts["temperature"] = False

    assert decide(VALID_SIGNALS, artifacts, CFG)[0] == "unavailable"


def test_upload_and_stub(monkeypatch):
    monkeypatch.setenv("LUCID_TRUST_STUB", "1")

    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)

    rng = np.random.default_rng(42)
    img = Image.fromarray(
        rng.integers(30, 220, (128, 128), dtype="uint8")
    )

    buf = io.BytesIO()
    img.save(buf, format="PNG")

    response = client.post(
        "/api/trust/assess",
        files={
            "file": (
                "demo.png",
                buf.getvalue(),
                "image/png",
            )
        },
    )

    assert response.status_code == 200
    assert response.json()["verdict"] == "unavailable"
    assert response.json()["signals"]["uncertainty"] is None

    response = client.post(
        "/api/trust/assess",
        files={
            "file": (
                "bad.txt",
                b"bad",
                "text/plain",
            )
        },
    )

    assert response.status_code == 422

from research.trust.nikhitha_trust_panel import service


def test_corrupted_config_returns_unavailable(tmp_path, monkeypatch):
    config_file = tmp_path / "trust_config.json"
    config_file.write_text("{invalid json")

    monkeypatch.setattr(service, "CONFIG_PATH", config_file)

    config, error = service._load_config()

    assert config is None
    assert error is not None


def test_non_object_config_returns_unavailable(tmp_path, monkeypatch):
    config_file = tmp_path / "trust_config.json"
    config_file.write_text("[]")

    monkeypatch.setattr(service, "CONFIG_PATH", config_file)

    config, error = service._load_config()

    assert config is None
    assert error is not None


def test_missing_config_threshold_returns_unavailable(tmp_path, monkeypatch):
    config_file = tmp_path / "trust_config.json"
    config_file.write_text('{"quality_low": 40}')

    monkeypatch.setattr(service, "CONFIG_PATH", config_file)

    config, error = service._load_config()

    assert config is None
    assert "quality_caution" in error

@pytest.mark.parametrize(
    "probabilities",
    [
        [0.7, 0.3],
        [0.5, 0.5],
    ],
)
def test_valid_probabilities(probabilities):
    assert service._valid_probabilities(probabilities)


@pytest.mark.parametrize(
    "probabilities",
    [
        [0.8, 0.8],
        [1.2, -0.2],
        [float("nan"), 0.5],
        [float("inf"), 0.0],
        [0.0],
        [],
        [True, False],
    ],
)
def test_invalid_probabilities(probabilities):
    assert not service._valid_probabilities(probabilities)


def test_compatible_artifact_metadata():
    metadata = {
        "checkpoint_sha256": "abc123",
        "splits_sha256": "test-split-1",
        "preprocessing_version": "1.0",
    }

    assert service._metadata_compatible(metadata, metadata)


def test_missing_splits_hash_is_incompatible():
    artifact = {
        "checkpoint_sha256": "abc123",
        "preprocessing_version": "1.0",
    }

    expected = {
        "checkpoint_sha256": "abc123",
        "splits_sha256": "test-split-1",
        "preprocessing_version": "1.0",
    }

    assert not service._metadata_compatible(artifact, expected)


def test_wrong_preprocessing_version_is_incompatible():
    artifact = {
        "checkpoint_sha256": "abc123",
        "splits_sha256": "test-split-1",
        "preprocessing_version": "2.0",
    }

    expected = {
        "checkpoint_sha256": "abc123",
        "splits_sha256": "test-split-1",
        "preprocessing_version": "1.0",
    }

    assert not service._metadata_compatible(artifact, expected)

def _post_test_image(client):
    rng = np.random.default_rng(42)
    img = Image.fromarray(
        rng.integers(30, 220, (128, 128), dtype="uint8")
    )

    buf = io.BytesIO()
    img.save(buf, format="PNG")

    return client.post(
        "/api/trust/assess",
        files={"file": ("demo.png", buf.getvalue(), "image/png")},
    )


def test_corrupted_config_does_not_crash_api(tmp_path, monkeypatch):
    config_file = tmp_path / "trust_config.json"
    config_file.write_text("{invalid json")

    monkeypatch.setattr(service, "CONFIG_PATH", config_file)
    monkeypatch.setenv("LUCID_TRUST_STUB", "1")

    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)

    response = _post_test_image(client)

    assert response.status_code == 200
    assert response.json()["verdict"] == "unavailable"
    assert response.json()["signals"]["quality_score"] is not None


def test_missing_threshold_does_not_crash_api(tmp_path, monkeypatch):
    config_file = tmp_path / "trust_config.json"
    config_file.write_text('{"quality_low": 40}')

    monkeypatch.setattr(service, "CONFIG_PATH", config_file)
    monkeypatch.setenv("LUCID_TRUST_STUB", "1")

    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)

    response = _post_test_image(client)

    assert response.status_code == 200
    assert response.json()["verdict"] == "unavailable"
