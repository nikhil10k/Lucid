
import math

LABELS = {
    "ok": "No reliability flags raised",
    "caution": "Caution: lower reliability",
    "low_reliability": "Low reliability: unlike training data or poor image",
    "unavailable": "Unavailable: trust artifacts not found",
}


def _valid_number(value, minimum=None, maximum=None):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False

    if not math.isfinite(value):
        return False

    if minimum is not None and value < minimum:
        return False

    if maximum is not None and value > maximum:
        return False

    return True


def _unavailable(reasons):
    return (
        "unavailable",
        LABELS["unavailable"],
        reasons,
    )


def decide(signals, artifacts, config):
    # Validate input structures
    if not isinstance(signals, dict):
        return _unavailable(["Invalid signal data"])

    if not isinstance(artifacts, dict):
        return _unavailable(["Invalid artifact status"])

    if not isinstance(config, dict):
        return _unavailable(["Trust configuration unavailable or invalid"])

    # Validate required configuration thresholds
    required_thresholds = (
        "quality_low",
        "quality_caution",
        "uncertainty_caution",
    )

    for key in required_thresholds:
        if key not in config:
            return _unavailable([
                f"Missing configuration threshold: {key}"
            ])

    quality_low = config["quality_low"]
    quality_caution = config["quality_caution"]
    uncertainty_caution = config["uncertainty_caution"]

    if (
        not _valid_number(quality_low, 0, 100)
        or not _valid_number(quality_caution, 0, 100)
        or not _valid_number(uncertainty_caution, 0)
        or quality_low > quality_caution
    ):
        return _unavailable([
            "Invalid trust configuration thresholds"
        ])

    # Validate all required signals
    required_signals = {
        "quality_score": (0, 100),
        "calibrated_confidence": (0, 1),
        "uncertainty": (0, None),
        "ood_score": (0, None),
    }

    reasons = []

    for key, (minimum, maximum) in required_signals.items():
        if not _valid_number(
            signals.get(key),
            minimum,
            maximum,
        ):
            reasons.append(f"Missing or invalid signal: {key}")

    if not isinstance(signals.get("ood_flag"), bool):
        reasons.append("Missing or invalid signal: ood_flag")

    # Validate required artifacts
    required_artifacts = (
        "temperature",
        "ensemble",
        "ood",
    )

    for key in required_artifacts:
        if artifacts.get(key) is not True:
            reasons.append(
                f"Unavailable or incompatible artifact: {key}"
            )

    # Never report a successful assessment with incomplete data
    if reasons:
        return _unavailable(reasons)

    # Calculate reliability verdict
    rank = 0

    quality_score = signals["quality_score"]
    uncertainty = signals["uncertainty"]

    if quality_score < quality_low:
        rank = 2
        reasons.append(
            f"Image quality score {quality_score} out of 100 (very low)"
        )

    elif quality_score < quality_caution:
        rank = 1
        reasons.append(
            f"Image quality score {quality_score} out of 100 (low)"
        )

    if signals["ood_flag"] is True:
        rank = 2
        reasons.append(
            "Image looks unlike the training data"
        )

    if uncertainty > uncertainty_caution:
        rank = max(rank, 1)
        reasons.append(
            "The ensemble members disagree"
        )

    key = [
        "ok",
        "caution",
        "low_reliability",
    ][rank]

    return (
        key,
        LABELS[key],
        reasons or ["No reliability flags raised"],
    )
