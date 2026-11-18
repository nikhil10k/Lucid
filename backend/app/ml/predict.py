from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from app.ml.common import build_model, eval_tf, PREPROC_VERSION
from app.config import settings


def _validate_checkpoint_metadata(ck):
    if not isinstance(ck, dict):
        raise ValueError("Checkpoint must contain a metadata dictionary")

    required = ["state", "arch", "classes", "img_size", "preprocessing_version"]
    missing = [key for key in required if key not in ck]

    if missing:
        raise ValueError(
            "Checkpoint is missing required metadata: " + ", ".join(missing)
        )

    if ck["classes"] != ["NORMAL", "PNEUMONIA"]:
        raise ValueError(
            "Expected class order ['NORMAL', 'PNEUMONIA']"
        )

    if ck["arch"] != "resnet18":
        raise ValueError("Expected resnet18 architecture")

    if ck["preprocessing_version"] != PREPROC_VERSION:
        raise ValueError("Preprocessing version mismatch")

    if (
        isinstance(ck["img_size"], bool)
        or not isinstance(ck["img_size"], int)
        or ck["img_size"] <= 0
    ):
        raise ValueError("Invalid checkpoint image size")


@lru_cache(maxsize=4)
def _load(checkpoint_path=None):
    path = Path(checkpoint_path or settings.model_path)

    if not path.is_file():
        raise FileNotFoundError(f"Missing approved model: {path}")

    ck = torch.load(path, map_location="cpu", weights_only=True)

    _validate_checkpoint_metadata(ck)

    model = build_model("resnet18", 2, False)
    model.load_state_dict(ck["state"])
    model.eval()

    feature_model = torch.nn.Sequential(*list(model.children())[:-1])

    return model, feature_model, eval_tf(ck["img_size"])


def _tensor(images, checkpoint_path=None):
    _, _, tf = _load(checkpoint_path)

    if not images:
        raise ValueError("Empty image list")

    for im in images:
        if im.dtype != np.uint8 or im.ndim != 2:
            raise ValueError("Expected uint8 grayscale arrays [H,W]")

    return torch.stack([tf(Image.fromarray(im)) for im in images])


def predict_logits(images, checkpoint_path=None):
    model, _, _ = _load(checkpoint_path)

    with torch.no_grad():
        return model(_tensor(images, checkpoint_path)).numpy()


def predict_proba(images, checkpoint_path=None):
    z = torch.from_numpy(predict_logits(images, checkpoint_path))
    return torch.softmax(z, 1).numpy()


def extract_features(images, checkpoint_path=None):
    _, features, _ = _load(checkpoint_path)

    with torch.no_grad():
        return features(_tensor(images, checkpoint_path)).flatten(1).numpy()