import argparse, json, subprocess, hashlib
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score
from .ensemble import ensemble_stats, predict_with_uncertainty
from .risk_coverage import risk_coverage
from research.trust.eval_signal import signal_error_auc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--records")
    ap.add_argument("--name", default="kermany_test")
    ap.add_argument("--export-val", action="store_true")
    a = ap.parse_args()
    rng = np.random.default_rng(42)
    if a.demo:
        y = rng.integers(0, 2, 600)
        latent = (2 * y - 1) * 1.4 + rng.normal(size=len(y)) * 1.6
        mp = 1 / (
            1 + np.exp(-(latent[:, None] + rng.normal(size=(len(y), 3))))
        )
        baseline = mp[:, 0]
    else:
        from PIL import Image
        from app.ml.predict import predict_proba

        S = json.loads(Path("data/processed/splits.json").read_text())
        if a.export_val and a.records:
            ap.error("Validation export uses frozen val only")
        records = (
            json.loads(Path(a.records).read_text())
            if a.records
            else S["val" if a.export_val else "test"]
        )
        y = np.array([r["label"] for r in records])
        parts = []
        base = []
        for start in range(0, len(records), 32):
            images = []
            for r in records[start : start + 32]:
                with Image.open(r["path"]) as im:
                    images.append(np.array(im.convert("L")))
            parts.append(predict_with_uncertainty(images)["member_probs"])
            base.append(predict_proba(images)[:, 1])
            print(start + len(images), "/", len(y))
        mp = np.concatenate(parts)
        baseline = np.concatenate(base)
    stats = ensemble_stats(mp)
    p = stats["mean_prob"]
    s = stats["std"]
    if a.export_val:
        dest = Path("private_inputs")
        dest.mkdir(exist_ok=True)
        hashes = [
            hashlib.sha256(
                Path(f"trust_artifacts/ensemble/seed{i}.pt").read_bytes()
            ).hexdigest()
            for i in (1, 2, 3)
        ]
        np.savez(
            dest / "ensemble_val.npz",
            std=s,
            split="val",
            source="kermany",
            ensemble_hashes=hashes,
        )
        print("Saved private_inputs/ensemble_val.npz")
        return
    # Same fixed baseline mistakes for comparing warning signals.
    wrong = ((baseline >= 0.5) != y).astype(int)
    ensemble_wrong = ((p >= 0.5) != y).astype(int)
    soft = 1 - np.maximum(baseline, 1 - baseline)
    result = dict(
        n=len(y),
        members=mp.shape[1],
        single_accuracy=accuracy_score(y, baseline >= 0.5),
        ensemble_accuracy=accuracy_score(y, p >= 0.5),
        single_auc=signal_error_auc(baseline, y),
        ensemble_auc=signal_error_auc(p, y),
        uncertainty_on_baseline_errors=signal_error_auc(s, wrong),
        uncertainty_on_ensemble_errors=signal_error_auc(s, ensemble_wrong),
        softmax_on_baseline_errors=signal_error_auc(soft, wrong),
        random_on_baseline_errors=signal_error_auc(rng.random(len(y)), wrong),
    )
    for name, correct in [
        ("single", 1 - wrong),
        ("ensemble", 1 - ensemble_wrong),
    ]:
        vals = [
            correct[rng.integers(0, len(y), len(y))].mean()
            for _ in range(500)
        ]
        result[name + "_accuracy_ci"] = np.percentile(
            vals, [2.5, 97.5]
        ).tolist()
    mode = "demo" if a.demo else a.name
    out = Path("results/trust/disha") / mode
    out.mkdir(parents=True, exist_ok=True)
    curve = risk_coverage(1 - wrong, s)
    softcurve = risk_coverage(1 - wrong, soft)
    curve["accuracy_softmax"] = softcurve["accuracy"]
    curve.to_csv(out / "risk_coverage.csv", index=False)
    pd.DataFrame(
        dict(
            label=y,
            baseline_prob=baseline,
            ensemble_prob=p,
            std=s,
            entropy=stats["entropy"],
        )
    ).to_csv(out / "per_image.csv", index=False)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for col in ["accuracy", "accuracy_softmax", "accuracy_random"]:
        plt.plot(curve.coverage, curve[col], label=col)
    plt.xlabel("Coverage retained")
    plt.ylabel("Fixed baseline accuracy")
    plt.title(f"{mode}; n={len(y)}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / "fig_risk_coverage.png", dpi=150)
    plt.close()
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True
    ).strip()
    meta = dict(
        author="disha",
        date=datetime.now(timezone.utc).isoformat(),
        git_commit=commit,
        seed=42,
        checkpoint="synthetic" if a.demo else "seeds 1,2,3",
        dataset=mode,
        n=len(y),
    )
    (out / "uncertainty.json").write_text(
        json.dumps(dict(meta=meta, results=result), indent=2)
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
