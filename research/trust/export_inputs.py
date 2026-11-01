
"""Export real OOD features using the existing frozen classifier.

Run from the repository root.
Private arrays are written to private_inputs/ and must not be committed.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from research.trust.niharika_ood.feature_adapter import extract_batch


def sha256_file(path):
    h = hashlib.sha256()

    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)

    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--split",
        choices=["train", "val", "test"],
        required=True,
    )

    parser.add_argument(
        "--out",
        required=True,
    )

    parser.add_argument(
        "--checkpoint",
        default="models/classifier.pt",
    )

    args = parser.parse_args()

    split_path = Path("data/processed/splits.json")
    checkpoint_path = Path(args.checkpoint)

    if not split_path.is_file():
        raise FileNotFoundError(split_path)

    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)

    splits = json.loads(split_path.read_text())

    if splits.get("classes") != ["NORMAL", "PNEUMONIA"]:
        raise ValueError("Unexpected class order; ask Nikhil")

    records = splits[args.split]

    if not records:
        raise ValueError(f"Empty {args.split} split")

    for record in records:
        if not Path(record["path"]).is_file():
            raise FileNotFoundError(
                "Missing image path: " + record["path"]
            )

        if record["label"] not in [0, 1]:
            raise ValueError("Unexpected label")

    features = []
    logits = []
    probabilities = []

    batch_size = 32

    for start in range(0, len(records), batch_size):
        batch_records = records[start:start + batch_size]

        images = []

        for record in batch_records:
            with Image.open(record["path"]) as image:
                images.append(image.convert("L").copy())

        output = extract_batch(images)

        features.append(output["features"])
        logits.append(output["logits"])
        probabilities.append(output["probs"])

        completed = start + len(batch_records)

        print(
            f"{args.split}: {completed}/{len(records)} images processed"
        )

    features = np.concatenate(features, axis=0)
    logits = np.concatenate(logits, axis=0)
    probabilities = np.concatenate(probabilities, axis=0)

    labels = np.array(
        [record["label"] for record in records],
        dtype=int,
    )

    if features.shape != (len(records), 512):
        raise ValueError(
            f"Expected features ({len(records)}, 512); "
            f"received {features.shape}"
        )

    if logits.shape != (len(records), 2):
        raise ValueError("Unexpected logits shape")

    if probabilities.shape != (len(records), 2):
        raise ValueError("Unexpected probability shape")

    if not np.isfinite(features).all():
        raise ValueError("Non-finite feature values")

    if not np.isfinite(probabilities).all():
        raise ValueError("Non-finite probabilities")

    if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-5):
        raise ValueError("Probability rows do not sum to one")

    payload = {
        "features": features,
        "logits": logits,
        "probs": probabilities,
        "labels": labels,
        "source": "kermany",
        "split": args.split,
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "splits_sha256": sha256_file(split_path),
    }

    destination = Path(args.out)
    destination.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(destination, **payload)

    print("\nExport completed.")
    print("Output:", destination)
    print("Split:", args.split)
    print("Images:", len(records))
    print("Feature shape:", features.shape)
    print("Logit shape:", logits.shape)
    print("Probability shape:", probabilities.shape)
    print("Checkpoint SHA256:", payload["checkpoint_sha256"])
    print("Splits SHA256:", payload["splits_sha256"])


if __name__ == "__main__":
    main()
