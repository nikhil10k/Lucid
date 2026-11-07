import numpy as np


def ensemble_stats(member_probs):
    mp = np.asarray(member_probs, float)  # [N,M], pneumonia probabilities
    if (
        mp.ndim != 2
        or not np.isfinite(mp).all()
        or (mp < 0).any()
        or (mp > 1).any()
    ):
        raise ValueError("Need finite member_probs [N,M] in [0,1]")
    p = mp.mean(1)
    q = np.clip(p, 1e-12, 1 - 1e-12)
    return dict(
        mean_prob=p,
        std=mp.std(1),
        entropy=-(q * np.log2(q) + (1 - q) * np.log2(1 - q)),
        member_probs=mp,
    )


def load_ensemble(paths):
    import torch
    from app.ml.common import build_model, PREPROC_VERSION

    loaded = []
    for path in paths:
        ck = torch.load(path, map_location="cpu", weights_only=True)
        if (
            ck["classes"] != ["NORMAL", "PNEUMONIA"]
            or ck["arch"] != "resnet18"
        ):
            raise ValueError("Unexpected ensemble architecture/class order")
        if ck["preprocessing_version"] != PREPROC_VERSION:
            raise ValueError("Preprocessing mismatch")
        m = build_model(ck["arch"], 2, False)
        m.load_state_dict(ck["state"])
        m.eval()
        loaded.append((m, ck))
    return loaded


_LOADED = None


def predict_with_uncertainty(images, paths=None):
    import torch
    from PIL import Image
    from app.ml.common import eval_tf

    global _LOADED
    if paths is not None:
        loaded = load_ensemble(paths)
    else:
        if _LOADED is None:
            _LOADED = load_ensemble(
                [f"trust_artifacts/ensemble/seed{i}.pt" for i in (1, 2, 3)]
            )
        loaded = _LOADED
    all_probs = []
    with torch.no_grad():
        for model, ck in loaded:
            tf = eval_tf(ck["img_size"])
            x = torch.stack([tf(Image.fromarray(im)) for im in images])
            all_probs.append(torch.softmax(model(x), 1)[:, 1].numpy())
    return ensemble_stats(np.stack(all_probs, axis=1))
