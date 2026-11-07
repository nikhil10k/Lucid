
import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .mahalanobis import *
from .baselines import max_softmax_score
from research.trust.eval_signal import signal_error_auc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true")

    for key in ["train", "val", "test", "external"]:
        ap.add_argument("--" + key)

    a = ap.parse_args()
    rng = np.random.default_rng(42)

    if a.demo:

        def sample(n, shift=0):
            y = rng.integers(0, 2, n)
            f = rng.normal(size=(n, 12)) + y[:, None] + shift
            p = np.clip(
                0.2 + 0.6 * y + rng.normal(size=n, scale=0.35),
                0.01,
                0.99,
            )
            return dict(
                features=f,
                labels=y,
                probs=np.stack([1 - p, p], 1),
            )

        tr, va, te, ext = (
            sample(600),
            sample(200),
            sample(200),
            sample(200, 3),
        )
        fingerprint = "synthetic"
        splits_fingerprint = "synthetic"

    else:
        if not all([a.train, a.val, a.test]):
            ap.error("Need --train --val --test")

        def load(path):
            with np.load(path, allow_pickle=False) as z:
                return dict(z)

        tr, va, te = map(load, [a.train, a.val, a.test])

        for data, split in [
            (tr, "train"),
            (va, "val"),
            (te, "test"),
        ]:
            if str(data["split"]) != split:
                raise ValueError("Wrong split")

        fingerprint = str(tr["checkpoint_sha256"])

        if any(
            str(d["checkpoint_sha256"]) != fingerprint
            for d in [va, te]
        ):
            raise ValueError("Checkpoint mismatch")

        splits_fingerprint = str(tr["splits_sha256"])

        if any(
            str(d["splits_sha256"]) != splits_fingerprint
            for d in [va, te]
        ):
            raise ValueError("Dataset split mismatch")

        ext = load(a.external) if a.external else None

        if (
            ext is not None
            and str(ext["checkpoint_sha256"]) != fingerprint
        ):
            raise ValueError("External checkpoint mismatch")

    params = fit_mahalanobis(tr["features"], tr["labels"])

    vs = mahalanobis_score(va["features"], params)
    threshold = threshold_at_percentile(vs)

    ts = mahalanobis_score(te["features"], params)

    wrong = (
        te["probs"].argmax(1) != te["labels"]
    ).astype(int)

    result = dict(
        threshold=threshold,
        val_flag_rate=float((vs > threshold).mean()),
        test_flag_rate=float((ts > threshold).mean()),
        test_error_auc=signal_error_auc(ts, wrong),
        test_softmax_error_auc=signal_error_auc(
            max_softmax_score(te["probs"]),
            wrong,
        ),
        external_status="pending" if ext is None else "evaluated",
    )

    if ext is not None:
        es = mahalanobis_score(ext["features"], params)

        target = np.r_[
            np.zeros(len(ts)),
            np.ones(len(es)),
        ]

        result["ood_auc"] = signal_error_auc(
            np.r_[ts, es],
            target,
        )

        soft = np.r_[
            max_softmax_score(te["probs"]),
            max_softmax_score(ext["probs"]),
        ]

        result["ood_softmax_auc"] = signal_error_auc(
            soft,
            target,
        )

        if np.isin(ext["labels"], [0, 1]).all():
            ew = (
                ext["probs"].argmax(1) != ext["labels"]
            ).astype(int)

            result["external_error_auc"] = signal_error_auc(
                es,
                ew,
            )

            result["external_softmax_error_auc"] = (
                signal_error_auc(
                    max_softmax_score(ext["probs"]),
                    ew,
                )
            )

            result["external_random_auc"] = signal_error_auc(
                rng.random(len(ew)),
                ew,
            )

        else:
            result["external_error_status"] = (
                "No disease labels; OOD sanity check only"
            )

    mode = (
        "demo"
        if a.demo
        else (
            "real_" + str(ext["source"])
            if ext is not None
            else "real"
        )
    )

    out = Path("results/trust/niharika") / mode
    out.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(
        dict(
            score=ts,
            wrong=wrong,
            flag=ts > threshold,
        )
    ).to_csv(out / "scores.csv", index=False)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.hist(
        ts,
        bins=25,
        alpha=0.6,
        label=f"ID test n={len(ts)}",
        density=True,
    )

    if ext is not None:
        plt.hist(
            es,
            bins=25,
            alpha=0.6,
            label=f"OOD n={len(es)}",
            density=True,
        )

    plt.axvline(
        threshold,
        color="black",
        linestyle="--",
        label="Validation cutoff",
    )

    plt.xlabel("Squared Mahalanobis score")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        out / "fig_score_distributions.png",
        dpi=150,
    )
    plt.close()

    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        text=True,
    ).strip()

    # Classify the experiment so that exploratory results
    # cannot be mistaken for approved benchmark results.
    if a.demo:
        experiment_type = "demo"
        experiment_note = (
            "Synthetic demonstration data; not a real evaluation."
        )
    elif ext is None:
        experiment_type = "id_only"
        experiment_note = "External dataset evaluation is pending."
    elif str(ext["source"]) == "non_chest_photos":
        experiment_type = "sanity_check"
        experiment_note = (
            f"{len(ext['features'])} non-chest photos; "
            "sanity check only, not a benchmark."
        )
    elif str(ext["source"]) == "kermany_noise_standin":
        experiment_type = "stand_in"
        experiment_note = (
            "Kermany noise stand-in; exploratory result, "
            "not an approved external benchmark."
        )
    else:
        experiment_type = "external_evaluation"
        experiment_note = "External dataset evaluation."

    meta = dict(
        author="niharika",
        date=datetime.now(timezone.utc).isoformat(),
        git_commit=commit,
        seed=42,
        checkpoint=fingerprint,
        splits_sha256=splits_fingerprint,
        dataset=mode,
        n=len(ts),  # Retained for backward compatibility.
        n_id=len(ts),
        n_external=0 if ext is None else len(ext["features"]),
        experiment_type=experiment_type,
        experiment_note=experiment_note,
    )

    (out / "ood.json").write_text(
        json.dumps(
            dict(meta=meta, results=result),
            indent=2,
        )
    )

    if not a.demo:
        dest = Path("trust_artifacts")
        dest.mkdir(exist_ok=True)

        np.savez(
            dest / "ood.npz",
            **params,
            threshold=threshold,
            fit_on="kermany_train",
            checkpoint_sha256=fingerprint,
            splits_sha256=splits_fingerprint,
        )

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
