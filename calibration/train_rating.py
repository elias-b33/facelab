"""Fit and evaluate a small, auditable human-rating predictor from local data.

No images, datasets, labels, or pretrained weights are downloaded. The input must
contain real panel ratings and the SAME raw feature extractor used in the app.
Synthetic fixtures are allowed only when explicitly marked; that mark travels
with the artifact so the app can refuse it for real people.

Requires NumPy. Run ``python calibration/train_rating.py --help`` for usage.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any

import numpy as np


CONTRACT_PATH = Path(__file__).with_name("rating_contract.json")
CONTRACT = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
FEATURE_KEYS = tuple(CONTRACT["featureKeys"])
FEATURE_SCHEMA = CONTRACT["featureSchema"]
DEFAULT_SEED = "facelab-rating-v1"


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def _text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a nonempty string")
    if len(value) > CONTRACT["maxStringLength"]:
        raise ValueError(f"{field} is too long")
    return value.strip()


def _number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise ValueError(f"{field} must be a finite number")
    try:
        converted = float(value)
    except OverflowError as error:
        raise ValueError(f"{field} must be a finite number") from error
    if not math.isfinite(converted):
        raise ValueError(f"{field} must be a finite number")
    return converted


def _integer(value: Any, field: str, minimum: int = 0) -> int:
    number = _number(value, field)
    if number != int(number) or number < minimum or number > 9_007_199_254_740_991:
        raise ValueError(f"{field} must be a safe integer >= {minimum}")
    return int(number)


def validate_manifest(manifest: Any) -> dict[str, Any]:
    if not isinstance(manifest, dict):
        raise ValueError("The manifest must be a JSON object")
    for key in ("datasetId", "datasetName", "sourceUrl", "license", "identityProtocol", "engineVersion"):
        _text(manifest.get(key), key)
    if not re.fullmatch(r"https?://[^\s/?#]+(?:[/?#][^\s]*)?", manifest["sourceUrl"], re.IGNORECASE):
        raise ValueError("sourceUrl must be an HTTP(S) source reference")
    if manifest.get("featureSchema") != FEATURE_SCHEMA:
        raise ValueError(f"featureSchema must be {FEATURE_SCHEMA}")
    if manifest.get("usage") not in ("research", "commercial-permitted"):
        raise ValueError("usage must be research or commercial-permitted; this is your source-rights declaration")
    if not isinstance(manifest.get("synthetic"), bool):
        raise ValueError("synthetic must explicitly be true or false")
    target = manifest.get("target")
    if not isinstance(target, dict):
        raise ValueError("target must describe the actual human-rating protocol")
    for key in ("label", "population", "protocol"):
        _text(target.get(key), f"target.{key}")
    low = _number(target.get("min"), "target.min")
    high = _number(target.get("max"), "target.max")
    if not 0 <= low < high <= 10:
        raise ValueError("Declare the original rating scale: 0 <= target.min < target.max <= 10")
    return manifest


def validate_rows(rows: Any, manifest: dict[str, Any]) -> list[dict[str, Any]]:
    """Validate labels/identity strictly; missing or weak features abstain later.

    V1 deliberately requires one front photograph per *verified person*. This
    prevents repeated images from inflating the calibration sample size. A
    caller-supplied ID is not proof of identity; identityProtocol records how the
    data curator checked this. Unknown SCUT image IDs must not be relabelled as
    verified people without a separate identity audit.
    """
    if not isinstance(rows, list) or not rows:
        raise ValueError("Input must contain JSONL rows")
    ids: set[str] = set()
    subjects: set[str] = set()
    low, high = manifest["target"]["min"], manifest["target"]["max"]
    for i, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"Row {i + 1} must be an object")
        record_id = _text(row.get("id"), f"row {i + 1} id")
        subject_id = _text(row.get("subject_id"), f"row {i + 1} subject_id")
        if record_id in ids:
            raise ValueError(f"Duplicate photo id: {record_id}")
        if subject_id in subjects:
            raise ValueError(f"Duplicate subject_id: {subject_id}. V1 needs one preselected front photo per person.")
        ids.add(record_id)
        subjects.add(subject_id)
        if row.get("identity_verified") is not True:
            raise ValueError(f"{record_id}: identity_verified must be true after a real identity audit")
        if row.get("view") != "front":
            raise ValueError(f"{record_id}: only front photographs are supported by this feature schema")
        if row.get("feature_schema") != FEATURE_SCHEMA:
            raise ValueError(f"{record_id}: mismatched feature_schema")
        if row.get("engine_version") != manifest["engineVersion"]:
            raise ValueError(f"{record_id}: engine_version differs from the manifest")
        if not isinstance(row.get("quality_pass"), bool):
            raise ValueError(f"{record_id}: quality_pass must be true or false")
        rating = row.get("rating")
        if not isinstance(rating, dict):
            raise ValueError(f"{record_id}: rating with mean, count, and scale is required")
        score = _number(rating.get("mean"), f"{record_id} rating.mean")
        if not low <= score <= high:
            raise ValueError(f"{record_id}: rating.mean is outside the declared source scale")
        count = _integer(rating.get("count"), f"{record_id} rating.count", 1)
        scale = rating.get("scale")
        if (not isinstance(scale, list) or len(scale) != 2
                or [_number(v, f"{record_id} rating.scale") for v in scale] != [low, high]):
            raise ValueError(f"{record_id}: do not mix source scales or silently remap labels")
        if count < CONTRACT["minimumRatersPerPhoto"]:
            raise ValueError(f"{record_id}: need at least {CONTRACT['minimumRatersPerPhoto']} independent ratings per photo")
        features = row.get("features")
        confidence = row.get("confidence")
        if not isinstance(features, dict) or not isinstance(confidence, dict):
            raise ValueError(f"{record_id}: features and confidence must be objects")
        for key in FEATURE_KEYS:
            # A null/missing measurement is a measured abstention, never zero.
            if features.get(key) is not None:
                value = _number(features[key], f"{record_id} features.{key}")
                if abs(value) > 100_000:
                    raise ValueError(f"{record_id}: feature {key} is outside the engine's numerical bound")
            if confidence.get(key) is not None:
                value = _number(confidence[key], f"{record_id} confidence.{key}")
                if not 0 <= value <= 1:
                    raise ValueError(f"{record_id}: feature confidence must be between 0 and 1")
    if len(rows) < CONTRACT["minimumSubjects"]["total"]:
        raise ValueError(f"Need at least {CONTRACT['minimumSubjects']['total']} verified people for this four-way pilot split")
    return rows


def split_subjects(rows: list[dict[str, Any]], seed: str = DEFAULT_SEED) -> dict[str, list[str]]:
    """Fixed, label-independent partitions. No ID appears in two sets."""
    _text(seed, "seed")
    ordered = sorted((row["subject_id"] for row in rows),
                     key=lambda subject: (hashlib.sha256((seed + "\0" + subject).encode()).hexdigest(), subject))
    count = len(ordered)
    minimum = CONTRACT["minimumSubjects"]
    validation = max(minimum["validation"], math.floor(count * .15))
    calibration = max(minimum["calibration"], math.floor(count * .15))
    test = max(minimum["test"], math.floor(count * .20))
    training = count - validation - calibration - test
    if training < minimum["training"]:
        raise ValueError("Not enough people for independent training, validation, calibration, and test sets")
    a, b, c = training, training + validation, training + validation + calibration
    return {"train": ordered[:a], "validation": ordered[a:b], "calibration": ordered[b:c], "test": ordered[c:]}


def quality_reasons(row: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    if not row["quality_pass"]:
        reasons.append("photo-quality")
    for key in FEATURE_KEYS:
        if row["features"].get(key) is None:
            reasons.append("missing-feature")
        if row["confidence"].get(key) is None or row["confidence"][key] < CONTRACT["quality"]["minimumFeatureConfidence"]:
            reasons.append("low-feature-confidence")
    return sorted(set(reasons))


def _arrays(rows: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    return (np.asarray([[row["features"][key] for key in FEATURE_KEYS] for row in rows], dtype=np.float64),
            np.asarray([row["rating"]["mean"] for row in rows], dtype=np.float64))


def _design(z: np.ndarray, basis: str) -> np.ndarray:
    return z if basis == "linear" else np.concatenate((z, z * z), axis=1)


def _fit(z: np.ndarray, y: np.ndarray, regularization: float, basis: str) -> tuple[float, np.ndarray]:
    design = _design(z, basis)
    x_mean, y_mean = design.mean(axis=0), float(y.mean())
    centered = design - x_mean
    # Intercept is unpenalized. N-scaled ridge keeps lambda meaningful across n.
    matrix = centered.T @ centered / len(y) + regularization * np.eye(design.shape[1])
    coefficients = np.linalg.solve(matrix, centered.T @ (y - y_mean) / len(y))
    intercept = y_mean - float(x_mean @ coefficients)
    if not np.isfinite(coefficients).all() or not math.isfinite(intercept):
        raise ValueError("Ridge fitting failed to produce finite coefficients")
    return intercept, coefficients


def _predict(z: np.ndarray, intercept: float, coefficients: np.ndarray,
             basis: str, low: float, high: float) -> np.ndarray:
    return np.clip(intercept + _design(z, basis) @ coefficients, low, high)


def _rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    result = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        result[order[start:end]] = (start + end - 1) / 2
        start = end
    return result


def _correlation(a: np.ndarray, b: np.ndarray) -> float | None:
    aa, bb = a - a.mean(), b - b.mean()
    denominator = float(np.linalg.norm(aa) * np.linalg.norm(bb))
    if denominator <= 1e-12:
        return None
    return max(-1.0, min(1.0, float(aa @ bb) / denominator))


def regression_metrics(target: np.ndarray, predicted: np.ndarray) -> dict[str, float | None]:
    absolute = np.abs(target - predicted)
    return {"mae": float(absolute.mean()), "rmse": float(np.sqrt(np.mean((target - predicted) ** 2))),
            "pearson": _correlation(target, predicted), "spearman": _correlation(_rank(target), _rank(predicted)),
            "medianAbsoluteError": float(np.median(absolute)),
            "p90AbsoluteError": float(np.sort(absolute)[math.ceil(.9 * len(absolute)) - 1])}


def conformal_quantile(residuals: np.ndarray, coverage: float) -> float:
    coverage = _number(coverage, "coverage")
    if not CONTRACT["coverageMin"] <= coverage <= CONTRACT["coverageMax"]:
        raise ValueError(f"coverage must be between {CONTRACT['coverageMin']} and {CONTRACT['coverageMax']}")
    values = np.asarray(residuals, dtype=np.float64)
    if values.ndim != 1 or not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Conformal residuals must be a finite nonnegative one-dimensional array")
    n = len(values)
    order = math.ceil((n + 1) * coverage)
    if n < CONTRACT["minimumSubjects"]["calibration"] or order > n:
        raise ValueError("Too few independent calibration people for a finite interval at this coverage")
    # Do not interpolate: finite-sample split conformal uses this order statistic.
    return float(np.sort(values)[order - 1])


def predict_artifact(artifact: dict[str, Any], features: dict[str, float]) -> dict[str, Any]:
    """Small reference implementation for Python/TypeScript parity tests.

    This does not qualify artifacts for product use; it intentionally permits
    explicitly synthetic artifacts for numerical tests. The app's importer owns
    deployment readiness, source declarations, and runtime photo-quality gates.
    """
    z = np.asarray([(features[f["key"]] - f["mean"]) / f["scale"] for f in artifact["features"]])
    if not np.isfinite(z).all() or np.max(np.abs(z)) > artifact["quality"]["maxAbsZ"]:
        return {"ok": False, "reason": "out-of-domain"}
    prediction = artifact["intercept"] + sum(f["linear"] * value + f["quadratic"] * value * value
                                             for f, value in zip(artifact["features"], z))
    low, high = artifact["target"]["min"], artifact["target"]["max"]
    prediction = max(low, min(high, prediction))
    q = artifact["calibration"]["absoluteErrorQuantile"]
    return {"ok": True, "value": float(prediction), "lower": max(low, prediction - q),
            "upper": min(high, prediction + q)}


def train_model(rows: list[dict[str, Any]], manifest: dict[str, Any],
                seed: str = DEFAULT_SEED, coverage: float = .9
                ) -> tuple[dict[str, Any], dict[str, Any], dict[str, list[str]]]:
    """Select on validation, calibrate once, evaluate once; never refit afterward."""
    validate_manifest(manifest)
    validate_rows(rows, manifest)
    coverage = _number(coverage, "coverage")
    if not CONTRACT["coverageMin"] <= coverage <= CONTRACT["coverageMax"]:
        raise ValueError(f"coverage must be between {CONTRACT['coverageMin']} and {CONTRACT['coverageMax']}")
    splits = split_subjects(rows, seed)
    by_subject = {row["subject_id"]: row for row in rows}
    split_rows = {name: [by_subject[subject] for subject in subjects] for name, subjects in splits.items()}
    quality_rows = {name: [row for row in part if not quality_reasons(row)] for name, part in split_rows.items()}
    minimum = CONTRACT["minimumSubjects"]
    if len(quality_rows["train"]) < minimum["training"]:
        raise ValueError(f"Need {minimum['training']} usable training people after photo/feature-quality abstentions")

    # These are the ONLY rows allowed to influence feature transforms or fitting.
    train_x, train_y = _arrays(quality_rows["train"])
    means, raw_scales = train_x.mean(axis=0), train_x.std(axis=0)
    # Constant features become inert in the fit. A unit denominator is explicit
    # and serializable; no epsilon division or invented missing measurement.
    scales = np.where(raw_scales <= 1e-12, 1.0, raw_scales)
    train_z = (train_x - means) / scales
    low, high = float(manifest["target"]["min"]), float(manifest["target"]["max"])
    split_audit: dict[str, Any] = {}
    accepted: dict[str, list[dict[str, Any]]] = {}
    for name, part in split_rows.items():
        accepted[name] = []
        rejected: list[dict[str, Any]] = []
        for row in part:
            reasons = quality_reasons(row)
            if not reasons:
                vector = np.asarray([row["features"][key] for key in FEATURE_KEYS])
                if name != "train" and np.max(np.abs((vector - means) / scales)) > CONTRACT["quality"]["maxAbsZ"]:
                    reasons = ["out-of-domain"]
            if reasons:
                rejected.append({"id": row["id"], "subject_id": row["subject_id"], "reasons": reasons})
            else:
                accepted[name].append(row)
        split_audit[name] = {
            "totalRows": len(part), "acceptedRows": len(accepted[name]), "unusableRows": len(rejected),
            "coverage": len(accepted[name]) / len(part),
            "reasons": dict(Counter(reason for item in rejected for reason in item["reasons"])),
            "excluded": rejected,
        }
        required = minimum["training" if name == "train" else name]
        if len(accepted[name]) < required:
            raise ValueError(f"Need {required} usable {name} people; got {len(accepted[name])}/{len(part)} after abstentions")

    val_x, val_y = _arrays(accepted["validation"])
    val_z = (val_x - means) / scales
    candidates: list[dict[str, Any]] = []
    best: tuple[tuple[float, float, int, float], float, np.ndarray, str, float] | None = None
    for basis in CONTRACT["basisCandidates"]:
        for regularization in CONTRACT["regularizationCandidates"]:
            intercept, coefficients = _fit(train_z, train_y, regularization, basis)
            predictions = _predict(val_z, intercept, coefficients, basis, low, high)
            metrics = regression_metrics(val_y, predictions)
            candidates.append({"basis": basis, "regularization": regularization, **metrics})
            # Ties prefer the smaller model, then stronger regularization.
            ordering = (float(metrics["mae"]), float(metrics["rmse"]), 0 if basis == "linear" else 1, -regularization)
            if best is None or ordering < best[0]:
                best = (ordering, intercept, coefficients, basis, regularization)
    assert best is not None
    _, intercept, coefficients, basis, regularization = best

    cal_x, cal_y = _arrays(accepted["calibration"])
    cal_pred = _predict((cal_x - means) / scales, intercept, coefficients, basis, low, high)
    quantile = conformal_quantile(np.abs(cal_y - cal_pred), coverage)

    test_x, test_y = _arrays(accepted["test"])
    test_pred = _predict((test_x - means) / scales, intercept, coefficients, basis, low, high)
    metrics = regression_metrics(test_y, test_pred)
    baseline = regression_metrics(test_y, np.full_like(test_y, float(train_y.mean())))
    lower, upper = np.maximum(low, test_pred - quantile), np.minimum(high, test_pred + quantile)
    interval_coverage = float(np.mean((test_y >= lower) & (test_y <= upper)))
    n_features = len(FEATURE_KEYS)
    linear = coefficients[:n_features]
    quadratic = coefficients[n_features:] if basis == "quadratic" else np.zeros(n_features)
    features = [{"key": key, "mean": float(means[i]), "scale": float(scales[i]),
                 "linear": float(linear[i]), "quadratic": float(quadratic[i]),
                 "min": float(train_x[:, i].min()), "max": float(train_x[:, i].max())}
                for i, key in enumerate(FEATURE_KEYS)]
    data_hash = hashlib.sha256(_canonical(sorted(rows, key=lambda row: row["id"]))).hexdigest()
    artifact: dict[str, Any] = {
        "schemaVersion": CONTRACT["schemaVersion"], "kind": CONTRACT["kind"], "featureSchema": FEATURE_SCHEMA,
        "modelId": "", "createdAt": datetime.now(timezone.utc).isoformat(),
        "target": {key: manifest["target"][key] for key in ("label", "min", "max", "population", "protocol")},
        "provenance": {**{key: manifest[key] for key in ("datasetId", "datasetName", "sourceUrl", "license", "usage",
                                                          "identityProtocol", "synthetic", "engineVersion")},
                       "dataSha256": data_hash, "splitSeed": seed},
        "features": features, "intercept": float(intercept),
        "calibration": {"method": "split-conformal-absolute-residual", "coverage": coverage,
                        "absoluteErrorQuantile": quantile, "subjects": len(cal_y),
                        "totalSubjects": len(splits["calibration"]),
                        "unusableRows": split_audit["calibration"]["unusableRows"]},
        "evaluation": {"testSubjects": len(test_y), "testRows": len(test_y),
                       "testTotalRows": len(splits["test"]), "testUnusableRows": split_audit["test"]["unusableRows"],
                       "testCoverage": split_audit["test"]["coverage"],
                       "mae": metrics["mae"], "rmse": metrics["rmse"], "pearson": metrics["pearson"],
                       "spearman": metrics["spearman"], "baselineMae": baseline["mae"],
                       "baselineRmse": baseline["rmse"], "intervalCoverage": interval_coverage,
                       "meanIntervalWidth": float(np.mean(upper - lower))},
        "quality": dict(CONTRACT["quality"]),
        "training": {"subjects": len(train_y), "totalSubjects": len(splits["train"]),
                     "unusableRows": split_audit["train"]["unusableRows"],
                     "validationSubjects": len(val_y), "validationTotalSubjects": len(splits["validation"]),
                     "regularization": regularization, "basis": basis},
    }
    identity_payload = {key: value for key, value in artifact.items() if key not in ("createdAt", "modelId")}
    artifact["modelId"] = "facelab-ridge-" + hashlib.sha256(_canonical(identity_payload)).hexdigest()[:20]

    limitations = [
        "Predicts a panel's mean rating of a particular frontal photo on its declared source scale.",
        "Geometry-only baseline; no appearance embedding, profile model, or universal beauty labels.",
        "Dataset/person identity, rights, and human-rating declarations are supplied by the data curator, not independently verified by this script.",
        "Calibration coverage is marginal for exchangeable, quality-accepted people under the same capture and rating protocol; it is not guaranteed per person, subgroup, or under distribution shift.",
        "Reported intervals concern mean panel ratings, not the opinions of individual viewers.",
        "One photo per person is required; repeated-photo stability and external-dataset evaluation remain necessary.",
        "Do not retune on these test labels. Changing seed after seeing scores makes the test exploratory.",
    ]
    if manifest["synthetic"]:
        limitations.insert(0, "SYNTHETIC TEST DATA: this artifact must never rate real faces or be described as trained on human judgments.")
    if len(test_y) < 200:
        limitations.append("Fewer than 200 accepted test people: pilot estimates are noisy; this is not broad product validation.")
    report = {
        "modelId": artifact["modelId"], "synthetic": manifest["synthetic"], "inputPeople": len(rows),
        "featureSchema": FEATURE_SCHEMA, "features": list(FEATURE_KEYS), "sourceScale": [low, high],
        "splitAudit": split_audit, "candidateValidation": candidates,
        "selected": {"basis": basis, "regularization": regularization},
        "evaluation": {**artifact["evaluation"], **{key: metrics[key] for key in ("medianAbsoluteError", "p90AbsoluteError")}},
        "betterThanMeanBaseline": metrics["mae"] < baseline["mae"],
        "limitations": limitations,
        "splitProtocol": "SHA256(seed + NUL + verified subject ID), ordered, label-independent; one front photo per person. Training-only transforms and weights; validation-only hyperparameter selection; calibration-only residual quantile; untouched test.",
        "trainingOutlierPolicy": "Fit all photo/feature-quality-accepted training rows. The fixed |z| gate applies to validation, calibration, test, and runtime; min/max are descriptive only.",
    }
    return artifact, report, splits


def _strict_json(text: str) -> Any:
    def invalid_constant(value: str) -> None:
        raise ValueError(f"Non-finite JSON constant {value} is not permitted")
    return json.loads(text, parse_constant=invalid_constant)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Joined raw features + human-label JSONL; one front photo per verified person")
    parser.add_argument("--manifest", required=True, type=Path, help="Dataset, feature-engine, source scale, rights, and rating-protocol JSON")
    parser.add_argument("--out", required=True, type=Path, help="Destination model JSON (no raw photos or subject IDs)")
    parser.add_argument("--report", required=True, type=Path, help="Evaluation JSON including local excluded IDs; keep private")
    parser.add_argument("--splits", required=True, type=Path, help="Private reproducible subject split assignment JSON")
    parser.add_argument("--seed", default=DEFAULT_SEED, help="Freeze this before examining held-out outcomes")
    parser.add_argument("--coverage", type=float, default=CONTRACT["defaultCoverage"], help="Nominal split-conformal coverage, default 0.9")
    args = parser.parse_args()
    outputs = [args.out.resolve(), args.report.resolve(), args.splits.resolve()]
    if len(set(outputs)) != 3 or any(path in (args.input.resolve(), args.manifest.resolve()) for path in outputs):
        parser.error("Output files must be distinct and must not overwrite the inputs")
    try:
        rows = [_strict_json(line) for line in args.input.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
        manifest = _strict_json(args.manifest.read_text(encoding="utf-8-sig"))
        artifact, report, splits = train_model(rows, manifest, args.seed, args.coverage)
        for path, content in ((args.out, artifact), (args.report, report), (args.splits, splits)):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(content, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    except (ValueError, OSError, json.JSONDecodeError, np.linalg.LinAlgError) as error:
        parser.exit(2, f"Rating training failed: {error}\n")
    summary = {"modelId": artifact["modelId"], "synthetic": artifact["provenance"]["synthetic"],
               "sourceScale": report["sourceScale"], "evaluation": artifact["evaluation"],
               "betterThanMeanBaseline": report["betterThanMeanBaseline"],
               "note": "Evaluation describes this held-out corpus only; no claim of universal looks accuracy."}
    print(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
