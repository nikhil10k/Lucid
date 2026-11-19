
import hashlib
import json
import logging
from pathlib import Path

from .verdict import decide

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = ROOT / "research" / "trust" / "trust_config.json"
ARTIFACT_DIR = ROOT / "trust_artifacts"


def _load_config():
    """Load and validate the trust configuration safely."""
    if not CONFIG_PATH.is_file():
        return None, "Trust configuration file is missing"

    try:
        config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        log.exception("Could not load trust configuration")
        return None, f"Trust configuration could not be read: {type(exc).__name__}"

    if not isinstance(config, dict):
        return None, "Trust configuration must be a JSON object"

    required = (
        "quality_low",
        "quality_caution",
        "uncertainty_caution",
    )

    for key in required:
        if key not in config:
            return None, f"Missing configuration threshold: {key}"

    return config, None


def _checkpoint_hash(model_path):
    """Return the checkpoint SHA256, or None if unavailable."""
    try:
        if not model_path.is_file():
            return None

        digest = hashlib.sha256()

        with model_path.open("rb") as file:
            for chunk in iter(lambda: file.read(1024 * 1024), b""):
                digest.update(chunk)

        return digest.hexdigest()

    except OSError:
        log.exception("Could not read model checkpoint")
        return None


def _unavailable(signals, artifacts, config, errors):
    """Build an unavailable response without crashing the API."""
    verdict, label, reasons = decide(signals, artifacts, config)

    reasons = list(reasons)
    reasons.extend(errors)

    return {
        "verdict": "unavailable",
        "label": label,
        "reasons": reasons,
        "signals": signals,
        "artifacts": artifacts,
        "disclaimer": (
            "Research and educational software only. Not a diagnosis."
        ),
    }


def assess_image(img, stub=False):
    from app.services.imaging import quality
    from app.config import settings

    signals = {
        "quality_score": None,
        "calibrated_confidence": None,
        "uncertainty": None,
        "ood_score": None,
        "ood_flag": None,
    }

    artifacts = {
        "temperature": False,
        "ensemble": False,
        "ood": False,
    }

    errors = []

    # Calculate image quality independently.
    try:
        result = quality(img)
        score = result.get("score")

        if (
            isinstance(score, (int, float))
            and not isinstance(score, bool)
            and 0 <= score <= 100
        ):
            signals["quality_score"] = float(score)
        else:
            errors.append("Image quality score is missing or invalid")

    except Exception:
        log.exception("Image quality calculation failed")
        errors.append("Image quality calculation failed")

    # Load configuration safely.
    config, config_error = _load_config()

    if config_error:
        errors.append(config_error)

    # Check the model checkpoint independently.
    model_path = Path(settings.model_path)
    fingerprint = _checkpoint_hash(model_path)

    if fingerprint is None:
        errors.append("Model checkpoint is unavailable")

    if stub:
        errors.append("Demo mode: real trust signals are unavailable")

    # Real signal calculations are deliberately not attempted until
    # the approved implementations and compatible artifacts are supplied.
    if not stub and fingerprint is not None:
        if not (ARTIFACT_DIR / "temperature.json").is_file():
            errors.append("Calibration artifact is unavailable")

        ensemble_files = [
            ARTIFACT_DIR / "ensemble" / f"seed{i}.pt"
            for i in (1, 2, 3)
        ]

        if not all(path.is_file() for path in ensemble_files):
            errors.append("Complete ensemble artifacts are unavailable")

        if not (ARTIFACT_DIR / "ood.npz").is_file():
            errors.append("OOD artifact is unavailable")

    # All required signals and artifacts must be valid before
    # a successful reliability verdict can be returned.
    return _unavailable(
        signals,
        artifacts,
        config,
        errors,
    )

def _valid_probabilities(values):
    """Check that class probabilities form a valid probability vector."""
    if hasattr(values, "tolist"):
        values = values.tolist()

    if not isinstance(values, (list, tuple)) or len(values) < 2:
        return False

    for value in values:
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not __import__("math").isfinite(value)
            or value < 0
            or value > 1
        ):
            return False

    return abs(sum(values) - 1.0) <= 1e-5


def _metadata_compatible(artifact_metadata, expected_metadata):
    """Require matching checkpoint, split and preprocessing metadata."""
    if not isinstance(artifact_metadata, dict):
        return False

    if not isinstance(expected_metadata, dict):
        return False

    required = (
        "checkpoint_sha256",
        "splits_sha256",
        "preprocessing_version",
    )

    for key in required:
        artifact_value = artifact_metadata.get(key)
        expected_value = expected_metadata.get(key)

        if not isinstance(artifact_value, str) or not artifact_value:
            return False

        if not isinstance(expected_value, str) or not expected_value:
            return False

        if artifact_value != expected_value:
            return False

    return True
