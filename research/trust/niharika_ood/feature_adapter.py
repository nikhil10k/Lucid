
"""OOD feature extraction using an explicitly selected classifier checkpoint."""

from pathlib import Path

import numpy as np
import torch
from PIL import Image

from app.services.inference_service import _load
from app.ml.common import PREPROC_VERSION, build_model, eval_tf


EXPECTED_CLASSES = ["NORMAL", "PNEUMONIA"]
EXPECTED_ARCH = "resnet18"

# Cache explicit checkpoints by their resolved file paths.
_MODEL_CACHE = {}


def _validate_checkpoint_metadata(checkpoint):
    """Reject checkpoints incompatible with the shared inference pipeline."""

    if not isinstance(checkpoint, dict):
        raise ValueError("Checkpoint must contain a metadata dictionary")

    required = [
        "state",
        "arch",
        "classes",
        "img_size",
        "preprocessing_version",
    ]

    missing = [
        key for key in required
        if key not in checkpoint
    ]

    if missing:
        raise ValueError(
            "Checkpoint is missing required metadata: "
            + ", ".join(missing)
        )

    if checkpoint["arch"] != EXPECTED_ARCH:
        raise ValueError(
            f"Expected {EXPECTED_ARCH}, found {checkpoint['arch']}"
        )

    if checkpoint["classes"] != EXPECTED_CLASSES:
        raise ValueError(
            f"Expected class order {EXPECTED_CLASSES}"
        )

    if checkpoint["preprocessing_version"] != PREPROC_VERSION:
        raise ValueError(
            "Preprocessing version mismatch: "
            f"expected {PREPROC_VERSION}, "
            f"found {checkpoint['preprocessing_version']}"
        )

    image_size = checkpoint["img_size"]

    if (
        isinstance(image_size, bool)
        or not isinstance(image_size, int)
        or image_size <= 0
    ):
        raise ValueError("Invalid checkpoint image size")


def _get_model(checkpoint_path=None):
    if checkpoint_path is None:
        # Preserve the application's normal default model-loading behaviour.
        loaded = _load()

        if loaded is None:
            raise RuntimeError(
                "Existing classifier checkpoint could not be loaded"
            )

        model, checkpoint = loaded

    else:
        # Load the exact checkpoint selected by the exporter.
        path = Path(checkpoint_path).resolve()

        if not path.is_file():
            raise FileNotFoundError(path)

        cache_key = str(path)

        if cache_key not in _MODEL_CACHE:
            checkpoint = torch.load(path, map_location="cpu")

            _validate_checkpoint_metadata(checkpoint)

            model = build_model(
                checkpoint["arch"],
                len(checkpoint["classes"]),
                pretrained=False,
            )
            model.load_state_dict(checkpoint["state"])
            model.eval()

            _MODEL_CACHE[cache_key] = (model, checkpoint)

        model, checkpoint = _MODEL_CACHE[cache_key]

    # Validate both the default and explicitly selected checkpoint paths.
    _validate_checkpoint_metadata(checkpoint)

    return model, checkpoint


def extract_batch(images, checkpoint_path=None):
    """Return logits, probabilities and 512-dimensional features."""

    model, checkpoint = _get_model(checkpoint_path)
    transform = eval_tf(checkpoint["img_size"])

    tensors = []

    for image in images:
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image.astype("uint8"))

        tensors.append(transform(image))

    if not tensors:
        raise ValueError("Empty image batch")

    batch = torch.stack(tensors)

    captured = {}

    def capture_features(module, inputs, output):
        captured["features"] = inputs[0].detach()

    handle = model.fc.register_forward_hook(capture_features)

    try:
        model.eval()

        with torch.no_grad():
            logits = model(batch)

    finally:
        handle.remove()

    features = captured["features"].cpu().numpy()
    logits = logits.cpu().numpy()

    shifted = logits - logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(shifted)
    probabilities = exp_logits / exp_logits.sum(
        axis=1, keepdims=True
    )

    if features.shape[1] != 512:
        raise ValueError(
            f"Expected 512 features, received {features.shape}"
        )

    return {
        "features": features,
        "logits": logits,
        "probs": probabilities,
    }


def extract_features(images, checkpoint_path=None):
    return extract_batch(images, checkpoint_path)["features"]


def predict_logits(images, checkpoint_path=None):
    return extract_batch(images, checkpoint_path)["logits"]


def predict_proba(images, checkpoint_path=None):
    return extract_batch(images, checkpoint_path)["probs"]
