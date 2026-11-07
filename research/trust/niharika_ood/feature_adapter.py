
"""OOD feature extraction using an explicitly selected classifier checkpoint."""

from pathlib import Path

import numpy as np
import torch
from PIL import Image

from app.services.inference_service import _load
from app.ml.common import build_model, eval_tf


# Cache explicit checkpoints by their resolved file paths.
_MODEL_CACHE = {}


def _get_model(checkpoint_path=None):
    if checkpoint_path is None:
        # Preserve the application's normal default model-loading behaviour.
        loaded = _load()

        if loaded is None:
            raise RuntimeError(
                "Existing classifier checkpoint could not be loaded"
            )
    else:
        # Load the exact checkpoint selected by the exporter.
        path = Path(checkpoint_path).resolve()

        if not path.is_file():
            raise FileNotFoundError(path)

        cache_key = str(path)

        if cache_key not in _MODEL_CACHE:
            checkpoint = torch.load(path, map_location="cpu")

            model = build_model(
                checkpoint["arch"],
                len(checkpoint["classes"]),
                pretrained=False,
            )
            model.load_state_dict(checkpoint["state"])
            model.eval()

            _MODEL_CACHE[cache_key] = (model, checkpoint)

        loaded = _MODEL_CACHE[cache_key]

    model, checkpoint = loaded

    if checkpoint["arch"] != "resnet18":
        raise ValueError(
            f"Expected resnet18, found {checkpoint['arch']}"
        )

    if checkpoint["classes"] != ["NORMAL", "PNEUMONIA"]:
        raise ValueError("Unexpected checkpoint class order")

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
