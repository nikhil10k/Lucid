import hashlib
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "backend"))

from app.ml.common import SplitDS, build_model, eval_tf


ROOT = Path(__file__).resolve().parents[3]
CHECKPOINT = ROOT / "models" / "classifier.pt"
SPLITS = ROOT / "data" / "processed" / "splits.json"
OUT = ROOT / "results" / "trust" / "chitthara"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def export_split(model, items, split_name, transform, device, fingerprint):
    loader = DataLoader(
        SplitDS(items, transform),
        batch_size=32,
        shuffle=False,
    )

    all_logits = []
    all_labels = []

    model.eval()
    with torch.no_grad():
        for images, labels in loader:
            logits = model(images.to(device))
            all_logits.append(logits.cpu().numpy())
            all_labels.append(labels.numpy())

    logits = np.concatenate(all_logits, axis=0)
    labels = np.concatenate(all_labels, axis=0)

    OUT.mkdir(parents=True, exist_ok=True)
    destination = OUT / f"{split_name}.npz"

    np.savez_compressed(
        destination,
        logits=logits,
        labels=labels,
        split=np.array(split_name),
        checkpoint_sha256=np.array(fingerprint),
    )

    print(f"Saved {destination}")
    print(f"  split: {split_name}, examples: {len(labels)}, logits shape: {logits.shape}")


def main():
    checkpoint = torch.load(CHECKPOINT, map_location="cpu")
    with open(SPLITS, "r", encoding="utf-8") as f:
        import json
        splits = json.load(f)

    device = torch.device("cpu")
    model = build_model(
        checkpoint["arch"],
        len(checkpoint["classes"]),
        pretrained=False,
    )
    model.load_state_dict(checkpoint["state"])
    model.to(device)

    fingerprint = sha256(CHECKPOINT)
    transform = eval_tf(checkpoint["img_size"])

    export_split(model, splits["val"], "val", transform, device, fingerprint)
    export_split(model, splits["test"], "test", transform, device, fingerprint)


if __name__ == "__main__":
    main()