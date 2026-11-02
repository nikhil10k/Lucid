import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .calibration import softmax, fit_temperature, report, reliability_diagram
from research.trust.eval_signal import signal_error_auc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--val")
    ap.add_argument("--test")
    ap.add_argument("--external")
    a = ap.parse_args()

    out = Path("results/trust/chitthara") / ("demo" if a.demo else "real")
    out.mkdir(parents=True, exist_ok=True)

    if a.demo:
        rng = np.random.default_rng(42)

        def sample():
            z = rng.normal(size=(4000, 2)) * 3
            y = (rng.random(len(z)) < softmax(z, 2)[:, 1]).astype(int)
            return z, y

        zv, yv = sample()
        zt, yt = sample()
        fingerprint = "synthetic"
    else:
        if not a.val or not a.test:
            ap.error("Need --val and --test input NPZs")

        with np.load(a.val) as v, np.load(a.test) as t:
            if str(v["split"]) != "val" or str(t["split"]) != "test":
                raise ValueError("Wrong split")
            if str(v["checkpoint_sha256"]) != str(t["checkpoint_sha256"]):
                raise ValueError("Checkpoint mismatch")

            fingerprint = str(v["checkpoint_sha256"])
            zv, yv = v["logits"], v["labels"]
            zt, yt = t["logits"], t["labels"]

    T = fit_temperature(zv, yv)
    before, after = report(zt, yt, 1), report(zt, yt, T)
    wrong = (zt.argmax(1) != yt).astype(int)

    result = dict(
        T=T,
        before=before,
        after=after,
        error_auc_before=signal_error_auc(
            1 - softmax(zt).max(1), wrong
        ),
        error_auc_after=signal_error_auc(
            1 - softmax(zt, T).max(1), wrong
        ),
        random_auc=signal_error_auc(
            np.random.default_rng(42).random(len(yt)), wrong
        ),
    )

    for name, temp in [("before", 1), ("after", T)]:
        reliability_diagram(
            softmax(zt, temp), yt, out / f"reliability_{name}.png"
        )

    if a.external:
        with np.load(a.external) as e:
            if str(e["checkpoint_sha256"]) != fingerprint:
                raise ValueError("External checkpoint mismatch")
            result["external"] = report(e["logits"], e["labels"], T)

    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True
    ).strip()
    meta = dict(
        author="chitthara",
        date=datetime.now(timezone.utc).isoformat(),
        git_commit=commit,
        seed=42,
        checkpoint=fingerprint,
        dataset="synthetic" if a.demo else "kermany_test",
        n=len(yt),
    )

    (out / "calibration.json").write_text(
        json.dumps(dict(meta=meta, results=result), indent=2)
    )

    if not a.demo:
        dest = Path("trust_artifacts")
        dest.mkdir(exist_ok=True)
        (dest / "temperature.json").write_text(
            json.dumps(
                dict(
                    T=T,
                    fit_on="kermany_val",
                    seed=42,
                    date=meta["date"],
                    checkpoint_sha256=fingerprint,
                ),
                indent=2,
            )
        )

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()