
import json
import random
from pathlib import Path

random.seed(42)

# Dataset folders
raw_dir = Path("data/raw")
test_dir = Path("data/test")
output_dir = Path("data/processed_new")

classes = ["NORMAL", "PNEUMONIA"]
extensions = {".jpg", ".jpeg", ".png"}

train_data = []
val_data = []
test_data = []

# Create training and validation splits from training images only
for label, class_name in enumerate(classes):
    class_dir = raw_dir / class_name

    if not class_dir.exists():
        raise FileNotFoundError(f"Missing folder: {class_dir}")

    images = sorted(
        p for p in class_dir.iterdir()
        if p.is_file() and p.suffix.lower() in extensions
    )

    random.shuffle(images)

    val_count = round(len(images) * 0.15)
    val_images = images[:val_count]
    train_images = images[val_count:]

    train_data.extend(
        {"path": str(p), "label": label} for p in train_images
    )

    val_data.extend(
        {"path": str(p), "label": label} for p in val_images
    )

# Keep the official test images completely separate
for label, class_name in enumerate(classes):
    class_dir = test_dir / class_name

    if not class_dir.exists():
        raise FileNotFoundError(f"Missing folder: {class_dir}")

    images = sorted(
        p for p in class_dir.iterdir()
        if p.is_file() and p.suffix.lower() in extensions
    )

    test_data.extend(
        {"path": str(p), "label": label} for p in images
    )

# Save the new split file without changing existing files
output_dir.mkdir(parents=True, exist_ok=True)

splits = {
    "classes": classes,
    "train": train_data,
    "val": val_data,
    "test": test_data,
}

output_file = output_dir / "splits.json"

with output_file.open("w", encoding="utf-8") as f:
    json.dump(splits, f, indent=2)

print("\nDataset splits created successfully!")
print("Classes:", classes)
print("Training images:", len(train_data))
print("Validation images:", len(val_data))
print("Official test images:", len(test_data))
print("Saved to:", output_file)

