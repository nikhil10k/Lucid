
"""Export model features and predictions for trust/OOD evaluation.

Run from the repository root.

Kermany exports use train/val/test splits from data/processed/splits.json.
External exports use a JSON list of records with path and label fields.
Private output arrays must not be committed.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image


def sha256_file(path):
    digest = hashlib.sha256()

    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(
        description="Export features and predictions for OOD evaluation."
    )
    parser.add_argument(
        "--records",
        help="Optional JSON file containing external image records.",
    )
    parser.add_argument(
        "--source",
        default="kermany",
        help="Dataset/source identifier for the exported records.",
    )
    parser.add_argument(
        "--split",
        choices=["train", "val", "test"],
        default="val",
        help="Kermany split to export when --records is not supplied.",
    )
    parser.add_argument("--out", required=True, help="Destination NPZ file.")
    parser.add_argument(
        "--checkpoint",
        default="models/classifier.pt",
        help="Exact checkpoint used for inference and hashing.",
    )
    args = parser.parse_args()

    from app.ml.predict import (
        predict_logits,
        predict_proba,
        extract_features,
    )
    from app.ml.common import PREPROC_VERSION
    from app.services.imaging import quality

    split_path = Path("data/processed/splits.json").resolve()
    checkpoint_path = Path(args.checkpoint).resolve()

    if not split_path.is_file():
        raise FileNotFoundError(f"Missing split manifest: {split_path}")

    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")

    splits = json.loads(split_path.read_text(encoding="utf-8"))
    classes = splits.get("classes")

    if classes != ["NORMAL", "PNEUMONIA"]:
        raise ValueError("Unexpected class order; ask Nikhil")

    external = args.records is not None

    if external:
        records_path = Path(args.records).resolve()

        if not records_path.is_file():
            raise FileNotFoundError(f"Missing records file: {records_path}")

        records = json.loads(records_path.read_text(encoding="utf-8"))
        source = args.source
        export_split = "external"
    else:
        records_path = None
        records = splits[args.split]
        source = "kermany"
        export_split = args.split

    if not isinstance(records, list) or not records:
        raise ValueError("Records must be a nonempty JSON list")

    labels = []

    for record in records:
        if not isinstance(record, dict) or "path" not in record:
            raise ValueError("Each record must contain a path")

        image_path = Path(record["path"])

        if not image_path.is_file():
            raise FileNotFoundError("Missing image path: " + str(image_path))

        if "label" not in record:
            raise ValueError("Each record must contain a label")

        label = record["label"]

        if isinstance(label, bool) or not isinstance(label, int):
            raise ValueError("Labels must be integers")

        allowed_labels = [-1, 0, 1] if external else [0, 1]

        if label not in allowed_labels:
            raise ValueError(
                f"Unexpected label {label} for source {source}"
            )

        labels.append(label)

    arrays = {
        "features": [],
        "logits": [],
        "probs": [],
        "quality": [],
    }

    batch_size = 32

    for start in range(0, len(records), batch_size):
        batch = records[start:start + batch_size]
        images = []

        for record in batch:
            with Image.open(record["path"]) as image:
                images.append(np.array(image.convert("L"), dtype=np.uint8))

        checkpoint_arg = str(checkpoint_path)

        arrays["logits"].append(
            predict_logits(images, checkpoint_path=checkpoint_arg)
        )
        arrays["probs"].append(
            predict_proba(images, checkpoint_path=checkpoint_arg)
        )
        arrays["features"].append(
            extract_features(images, checkpoint_path=checkpoint_arg)
        )
        arrays["quality"].extend(
            quality(image)["score"] for image in images
        )

        completed = start + len(batch)
        print(f"{source}: {completed}/{len(records)} images processed")

    features = np.concatenate(arrays["features"], axis=0)
    logits = np.concatenate(arrays["logits"], axis=0)
    probabilities = np.concatenate(arrays["probs"], axis=0)
    quality_scores = np.asarray(arrays["quality"], dtype=float)
    labels = np.asarray(labels, dtype=int)

    count = len(records)

    if features.shape != (count, 512):
        raise ValueError(f"Expected features ({count}, 512); got {features.shape}")

    if logits.shape != (count, 2):
        raise ValueError(f"Expected logits ({count}, 2); got {logits.shape}")

    if probabilities.shape != (count, 2):
        raise ValueError(
            f"Expected probabilities ({count}, 2); got {probabilities.shape}"
        )

    if quality_scores.shape != (count,):
        raise ValueError("Unexpected quality-score shape")

    for name, values in (
        ("features", features),
        ("logits", logits),
        ("probabilities", probabilities),
        ("quality scores", quality_scores),
    ):
        if not np.isfinite(values).all():
            raise ValueError(f"Non-finite {name}")

    if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-5):
        raise ValueError("Probability rows do not sum to one")

    checkpoint_hash = sha256_file(checkpoint_path)
    splits_hash = sha256_file(split_path)

    payload = {
        "features": features,
        "logits": logits,
        "probs": probabilities,
        "quality": quality_scores,
        "labels": labels,
        "source": source,
        "split": export_split,
        "checkpoint_sha256": checkpoint_hash,
        "classes": np.asarray(classes),
        "preprocessing_version": PREPROC_VERSION,
    }

    if external:
        payload["reference_splits_sha256"] = splits_hash
        payload["records_sha256"] = sha256_file(records_path)
    else:
        payload["splits_sha256"] = splits_hash

    destination = Path(args.out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination, **payload)

    print("\nExport completed.")
    print("Output:", destination)
    print("Source:", source)
    print("Split:", export_split)
    print("Images:", count)
    print("Feature shape:", features.shape)
    print("Checkpoint:", checkpoint_path)
    print("Checkpoint SHA256:", checkpoint_hash)

    if external:
        print("Records SHA256:", payload["records_sha256"])
    else:
        print("Splits SHA256:", splits_hash)


if __name__ == "__main__":
    main()
