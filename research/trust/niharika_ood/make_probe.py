"""Private probe inputs. Photos have label -1: no disease label is invented."""

import argparse, hashlib, json
from pathlib import Path
import numpy as np
from PIL import Image
from .feature_adapter import predict_proba, extract_features
from app.config import settings


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["noise", "photos"], required=True)
    a = ap.parse_args()
    rng = np.random.default_rng(42)
    if a.mode == "noise":
        S = json.loads(Path("data/processed/splits.json").read_text())
        records = S["test"]
        labels = np.array([r["label"] for r in records])

        def get_images():
            for r in records:
                with Image.open(r["path"]) as im:
                    gray = np.array(im.convert("L"))
                yield np.clip(
                    gray.astype(float) + rng.normal(0, 35, gray.shape), 0, 255
                ).astype("uint8")

        images = list(get_images())
        source = "kermany_noise_standin"
    else:
        from sklearn.datasets import load_sample_images

        images = [
            np.array(Image.fromarray(im).convert("L"))
            for im in load_sample_images().images
        ]
        labels = np.full(len(images), -1, int)
        source = "non_chest_photos"
    fs = []
    ps = []
    for i in range(0, len(images), 32):
        fs.append(extract_features(images[i : i + 32]))
        ps.append(predict_proba(images[i : i + 32]))
    dest = Path("private_inputs")
    dest.mkdir(exist_ok=True)
    h = hashlib.sha256(Path(settings.model_path).read_bytes()).hexdigest()
    np.savez_compressed(
        dest / f"{source}.npz",
        features=np.concatenate(fs),
        probs=np.concatenate(ps),
        labels=labels,
        source=source,
        split="test",
        checkpoint_sha256=h,
    )
    print(dest / f"{source}.npz")


if __name__ == "__main__":
    main()