"""OOD feature extraction using the existing trained ResNet18 checkpoint."""

import numpy as np
import torch
from PIL import Image

from app.services.inference_service import _load
from app.ml.common import eval_tf


def _get_model():
    loaded = _load()

    if loaded is None:
        raise RuntimeError("Existing classifier checkpoint could not be loaded")

    model, checkpoint = loaded

    if checkpoint["arch"] != "resnet18":
        raise ValueError(
            f"Expected resnet18, found {checkpoint['arch']}"
        )

    if checkpoint["classes"] != ["NORMAL", "PNEUMONIA"]:
        raise ValueError("Unexpected checkpoint class order")

    return model, checkpoint


def extract_batch(images):
    """Return logits, probabilities and 512-dimensional features."""

    model, checkpoint = _get_model()
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


def extract_features(images):
    return extract_batch(images)["features"]


def predict_logits(images):
    return extract_batch(images)["logits"]


def predict_proba(images):
    return extract_batch(images)["probs"]