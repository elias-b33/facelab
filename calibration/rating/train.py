"""Train and audit a small, research-only supervised face-rating baseline.

Inputs are the shared TypeScript extractor's JSONL features and actual human
labels. No heuristic or synthetic labels, population remapping, implicit label
rescaling, test-set fitting, or learned protected attributes are used.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path

import numpy as np

FEATURE_SCHEMA = "facelab-geometry-v1"
MODEL_FORMAT = "facelab-rating-linear-v1"
SEED = 20261022
ALPHAS = (.01, .1, 1.0, 10.0, 100.0, 1000.0)
POLICY = {"minCoverage": .9, "maxAbsZ": 4.0, "maxRmsZ": 2.5}
SPLIT_NAMES = ("fit", "tuning", "calibration", "test")


def is_number(value) -> bool:
    return not isinstance(value, bool) and isinstance(value, (float, int)) and math.isfinite(value)


def validate_records(records: list[dict]) -> tuple[float, float]:
    if not records:
        raise ValueError("No feature records supplied")
    ids = set()
    targets, datasets = set(), set()
    for row in records:
        if not isinstance(row.get("id"), str) or not row["id"] or row["id"] in ids:
            raise ValueError("Every capture requires a distinct nonempty ID")
        ids.add(row["id"])
        if not isinstance(row.get("subjectId"), str) or not row["subjectId"]:
            raise ValueError("Every capture requires a verified within-dataset subjectId")
        features = row.get("features", {})
        if features.get("schemaVersion") != FEATURE_SCHEMA:
            raise ValueError("Incompatible feature schema; use the shared TypeScript extractor")
        if not isinstance(features.get("values"), dict) or not isinstance(features.get("quality"), dict):
            raise ValueError("Missing feature values or capture-quality provenance")
        for key, value in features["values"].items():
            if not key.startswith(("legacy.", "harmony.")) or value is not None and not is_number(value):
                raise ValueError("Features must be finite observed geometry or explicit nulls")
        label = row.get("label", {})
        if label.get("kind") != "mean-human-photo-rating" or label.get("scope") != "neutral_front_only":
            raise ValueError("Only documented human frontal-photo labels are accepted")
        if any(not is_number(label.get(k)) for k in ("mean", "min", "max", "count")):
            raise ValueError("Invalid human label metadata")
        if not label["min"] <= label["mean"] <= label["max"] or label["min"] >= label["max"] or label["count"] < 3:
            raise ValueError("Human rating is outside its documented source scale or lacks raters")
        source = row.get("source", {})
        if not all(source.get(k) for k in ("url", "license", "humanRatingsSha256")):
            raise ValueError("Source, licence and human-label hash evidence are required")
        targets.add((label["min"], label["max"]))
        datasets.add((row.get("dataset"), source["url"], source["license"], source["humanRatingsSha256"]))
    if len(targets) != 1 or len(datasets) != 1:
        raise ValueError("Train each documented source scale and cohort separately")
    return next(iter(targets))


def assign_subject_splits(records: list[dict], seed: int) -> dict[str, str]:
    """Split by person before feature filtering; all views/captures stay together."""
    subjects = sorted({r["subjectId"] for r in records})
    if len(subjects) < 60:
        raise ValueError("At least 60 distinct people required for the four-way research split")
    order = np.random.default_rng(seed).permutation(len(subjects))
    sizes = [math.floor(len(subjects) * p) for p in (.5, .15, .175)]
    sizes.append(len(subjects) - sum(sizes))
    assignment, offset = {}, 0
    for name, size in zip(SPLIT_NAMES, sizes):
        for index in order[offset:offset + size]:
            assignment[subjects[int(index)]] = name
        offset += size
    return assignment


def usable(row: dict) -> bool:
    features = row.get("features", {})
    quality = features.get("quality", {})
    return (features.get("schemaVersion") == FEATURE_SCHEMA
            and quality.get("frontUsable") is True
            and isinstance(quality.get("blockingReasons"), list)
            and len(quality["blockingReasons"]) == 0)


def subject_weights(rows: list[dict]) -> np.ndarray:
    counts = Counter(r["subjectId"] for r in rows)
    return np.asarray([1 / counts[r["subjectId"]] for r in rows], dtype=float)


def fit_transform(rows: list[dict]) -> tuple[list[dict], list[str]]:
    """Choose columns and normalization using fit subjects only."""
    keys = sorted(set().union(*(r["features"]["values"] for r in rows)))
    weights = subject_weights(rows)
    features, excluded = [], []
    for key in keys:
        raw_values = [r["features"]["values"].get(key) for r in rows]
        values = np.asarray([value if is_number(value) else np.nan for value in raw_values], dtype=float)
        observed = np.isfinite(values)
        availability = float(weights[observed].sum() / weights.sum())
        if availability < .9:
            excluded.append(key)
            continue
        mean = float(np.average(values[observed], weights=weights[observed]))
        scale = float(np.sqrt(np.average((values[observed] - mean) ** 2, weights=weights[observed])))
        if scale < 1e-8:
            excluded.append(key)
            continue
        features.append({"key": key, "mean": mean, "scale": scale, "weight": 0.0, "missingWeight": 0.0})
    if not features:
        raise ValueError("No sufficiently observed, nonconstant training geometry features")
    return features, excluded


def vectorize(row: dict, features: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    values = row["features"]["values"]
    z, missing = [], []
    for feature in features:
        value = values.get(feature["key"])
        absent = not is_number(value)
        z.append(0.0 if absent else (value - feature["mean"]) / feature["scale"])
        missing.append(float(absent))
    return np.asarray(z), np.asarray(missing)


def policy_check(row: dict, features: list[dict], policy: dict) -> tuple[bool, list[str]]:
    if not usable(row):
        return False, ["capture_quality"] + row["features"]["quality"].get("blockingReasons", [])
    z, missing = vectorize(row, features)
    observed = z[missing == 0]
    reasons = []
    if len(observed) / len(features) < policy["minCoverage"]:
        reasons.append("insufficient_feature_coverage")
    if len(observed) and float(np.max(np.abs(observed))) > policy["maxAbsZ"]:
        reasons.append("out_of_training_range")
    if len(observed) and float(np.sqrt(np.mean(observed ** 2))) > policy["maxRmsZ"]:
        reasons.append("out_of_training_distribution")
    return not reasons, reasons


def design(rows: list[dict], features: list[dict]) -> np.ndarray:
    return np.asarray([np.concatenate(([1.0], *vectorize(row, features))) for row in rows])


def fit_ridge(rows: list[dict], features: list[dict], penalty: float) -> np.ndarray:
    matrix = design(rows, features)
    y = np.asarray([r["label"]["mean"] for r in rows])
    weights = subject_weights(rows)
    regularizer = np.eye(matrix.shape[1]) * penalty
    regularizer[0, 0] = 0
    return np.linalg.solve(matrix.T @ (weights[:, None] * matrix) + regularizer, matrix.T @ (weights * y))


def predict(rows: list[dict], features: list[dict], coefficients: np.ndarray, bounds: tuple) -> np.ndarray:
    return np.clip(design(rows, features) @ coefficients, *bounds)


def average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    result = np.empty(len(values), dtype=float)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and values[order[j]] == values[order[i]]:
            j += 1
        result[order[i:j]] = (i + j - 1) / 2
        i = j
    return result


def metrics(target: np.ndarray, prediction: np.ndarray) -> dict:
    if not len(target):
        return {"subjects": 0, "mae": None, "rmse": None, "spearman": None}
    rho = None
    if len(target) > 1 and len(set(target)) > 1 and len(set(prediction)) > 1:
        rho = float(np.corrcoef(average_ranks(target), average_ranks(prediction))[0, 1])
    return {"subjects": len(target), "mae": float(np.mean(np.abs(target - prediction))),
            "rmse": float(np.sqrt(np.mean((target - prediction) ** 2))), "spearman": rho}


def subject_arrays(rows: list[dict], prediction: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    grouped = defaultdict(list)
    for row, value in zip(rows, prediction):
        grouped[row["subjectId"]].append((row["label"]["mean"], float(value)))
    means = [np.mean(values, axis=0) for _, values in sorted(grouped.items())]
    return (np.asarray([v[0] for v in means]), np.asarray([v[1] for v in means]))


def conformal_radius(residuals: list[float], coverage: float) -> float:
    if not 0 < coverage < 1 or not residuals or any(not is_number(x) or x < 0 for x in residuals):
        raise ValueError("Conformal calibration requires finite nonnegative residuals and coverage in (0,1)")
    order = math.ceil((len(residuals) + 1) * coverage - 1e-12)
    if order > len(residuals):
        raise ValueError("Too few accepted independent calibration subjects for a finite interval at this coverage")
    return float(sorted(residuals)[order - 1])


def bootstrap(target: np.ndarray, prediction: np.ndarray, baseline: np.ndarray, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    draws, ranks = [], []
    for _ in range(2000):
        take = rng.integers(0, len(target), len(target))
        error = np.abs(target[take] - prediction[take]).mean()
        gain = np.abs(target[take] - baseline[take]).mean() - error
        draws.append((error, gain))
        rho = metrics(target[take], prediction[take])["spearman"]
        if rho is not None:
            ranks.append(rho)
    interval = np.quantile(np.asarray(draws), [.025, .975], axis=0)
    return {"replicates": 2000, "mae95": interval[:, 0].tolist(), "maeImprovementOverMean95": interval[:, 1].tolist(),
            "spearman95": np.quantile(ranks, [.025, .975]).tolist() if ranks else None,
            "scope": "Person bootstrap conditional on this fixed model, split, rater panel and acceptance rule; not external-cohort evidence."}


def train(records: list[dict], seed: int = SEED, coverage: float = .9) -> tuple[dict, dict]:
    bounds = validate_records(records)
    assignments = assign_subject_splits(records, seed)
    partitions = {name: [r for r in records if assignments[r["subjectId"]] == name] for name in SPLIT_NAMES}
    fit_rows = [r for r in partitions["fit"] if usable(r)]
    if len({r["subjectId"] for r in fit_rows}) < 20:
        raise ValueError("Too few fit subjects pass capture quality")
    features, excluded_features = fit_transform(fit_rows)
    tuning = [r for r in partitions["tuning"] if policy_check(r, features, POLICY)[0]]
    if len({r["subjectId"] for r in tuning}) < 5:
        raise ValueError("Too few independent tuning subjects pass the predeclared policy")
    candidates = []
    for penalty in ALPHAS:
        coefficients = fit_ridge(fit_rows, features, penalty)
        target, prediction = subject_arrays(tuning, predict(tuning, features, coefficients, bounds))
        candidates.append((metrics(target, prediction)["mae"], penalty, coefficients))
    tuning_mae, penalty, coefficients = min(candidates, key=lambda item: (item[0], -item[1]))
    mean_target, _ = subject_arrays(fit_rows, np.zeros(len(fit_rows)))
    mean_baseline = float(mean_target.mean())
    calibration = [r for r in partitions["calibration"] if policy_check(r, features, POLICY)[0]]
    calibration_predictions = predict(calibration, features, coefficients, bounds) if calibration else []
    residuals = defaultdict(list)
    for row, prediction in zip(calibration, calibration_predictions):
        residuals[row["subjectId"]].append(abs(row["label"]["mean"] - float(prediction)))
    # A person's repeated views count once and calibrate their maximum error.
    radius = conformal_radius([max(v) for v in residuals.values()], coverage)
    test = partitions["test"]
    accepted = [r for r in test if policy_check(r, features, POLICY)[0]]
    if len({r["subjectId"] for r in accepted}) < 3:
        raise ValueError("Too few accepted test subjects to report independent evaluation")
    test_predictions = predict(accepted, features, coefficients, bounds)
    y, p = subject_arrays(accepted, test_predictions)
    baseline = np.full(len(y), mean_baseline)
    model_metrics, baseline_metrics = metrics(y, p), metrics(y, baseline)
    coverage_by_subject, intervals = defaultdict(list), []
    for row, prediction in zip(accepted, test_predictions):
        low, high = max(bounds[0], float(prediction) - radius), min(bounds[1], float(prediction) + radius)
        covered = low <= row["label"]["mean"] <= high
        coverage_by_subject[row["subjectId"]].append(covered)
        intervals.append({"id": row["id"], "subjectId": row["subjectId"], "target": row["label"]["mean"],
                          "prediction": float(prediction), "interval": [low, high], "covered": covered})
    test_people = len({r["subjectId"] for r in test})
    accepted_fraction = len(coverage_by_subject) / test_people
    observed_coverage = float(np.mean([all(v) for v in coverage_by_subject.values()]))
    for i, feature in enumerate(features):
        feature["weight"] = float(coefficients[1 + i])
        feature["missingWeight"] = float(coefficients[1 + len(features) + i])
    source = records[0]["source"]
    counts = {name: len({r["subjectId"] for r in partitions[name]}) for name in SPLIT_NAMES}
    model = {
        "format": MODEL_FORMAT, "id": f"{records[0]['dataset']}-geometry-research-{seed}",
        "featureSchema": FEATURE_SCHEMA, "researchOnly": True,
        "target": {"name": f"Mean human rating of a neutral frontal photograph (original {bounds[0]:g}–{bounds[1]:g} scale)",
                   "min": bounds[0], "max": bounds[1], "kind": "mean-human-photo-rating"},
        "intercept": float(coefficients[0]), "features": features, "policy": dict(POLICY),
        "calibration": {"method": "split-conformal-absolute", "coverage": coverage, "radius": radius, "subjects": len(residuals)},
        "evaluation": {"testSubjects": test_people, "mae": model_metrics["mae"], "baselineMae": baseline_metrics["mae"],
                       "spearman": model_metrics["spearman"], "intervalCoverage": observed_coverage, "acceptedFraction": accepted_fraction},
        "training": {"dataset": records[0]["dataset"], "sourceUrl": source["url"], "license": source["license"],
                     "fitSubjects": len({r["subjectId"] for r in fit_rows}), "tuningSubjects": len({r["subjectId"] for r in tuning}),
                     "calibrationSubjects": len(residuals), "testSubjects": test_people, "splitSeed": seed},
    }
    selective = []
    for threshold in (2.0, 3.0, 4.0, 6.0):
        policy = {**POLICY, "maxAbsZ": threshold}
        subset = [r for r in test if policy_check(r, features, policy)[0]]
        if subset:
            target, prediction = subject_arrays(subset, predict(subset, features, coefficients, bounds))
            selective.append({"maxAbsZ": threshold, "acceptedFraction": len(target) / test_people,
                              "model": metrics(target, prediction), "meanBaseline": metrics(target, np.full(len(target), mean_baseline))})
        else:
            selective.append({"maxAbsZ": threshold, "acceptedFraction": 0, "model": metrics(np.array([]), np.array([]))})
    report = {
        "format": "facelab-rating-evaluation-v1", "researchOnly": True, "productionAccuracyEstablished": False,
        "source": source, "sourceScale": list(bounds), "featureSchema": FEATURE_SCHEMA,
        "sourceSubjects": len(assignments), "sourceCaptures": len(records), "splitSeed": seed,
        "engineVersions": sorted({r["engineVersion"] for r in records if r.get("engineVersion")}),
        "captureProvenance": [{"id": r["id"], "subjectId": r["subjectId"],
                               "detector": r.get("detector"), "engineVersion": r.get("engineVersion"),
                               "pixelGatesEvaluated": r.get("pixelGatesEvaluated") is True,
                               "detectionFailure": r.get("detectionFailure")} for r in records],
        "splitSubjectsBeforeQualityFilter": counts,
        "splitIds": {name: sorted(k for k, value in assignments.items() if value == name) for name in SPLIT_NAMES},
        "qualityFailures": [{"id": r["id"], "subjectId": r["subjectId"], "split": assignments[r["subjectId"]],
                             "reasons": r["features"]["quality"].get("blockingReasons", [])} for r in records if not usable(r)],
        "selectedFeatures": [f["key"] for f in features], "excludedFeaturesFromFitOnly": excluded_features,
        "ridgePenalty": penalty, "tuningMae": tuning_mae,
        "tuningGrid": [{"penalty": a, "mae": error} for error, a, _ in candidates],
        "meanBaselineFromFitSubjects": mean_baseline,
        "testAccepted": model_metrics, "testMeanBaselineSameSubjects": baseline_metrics,
        "testAcceptedFraction": accepted_fraction, "testIntervalCoverage": observed_coverage,
        "intervalNominalCoverage": coverage, "intervalRadius": radius,
        "intervalMeanWidth": float(np.mean([r["interval"][1] - r["interval"][0] for r in intervals])),
        "calibrationAcceptedSubjects": len(residuals), "testPredictions": intervals,
        "testAbstentions": [{"id": r["id"], "subjectId": r["subjectId"], "reasons": policy_check(r, features, POLICY)[1]} for r in test if not policy_check(r, features, POLICY)[0]],
        "bootstrap": bootstrap(y, p, baseline, seed + 1),
        "selectiveDiagnostics": selective,
        "method": "Person-separated 50/15/17.5/remainder fit/tuning/calibration/test split fixed before quality filtering. Columns, normalization and ridge fits use fit subjects only. Tuning selects ridge penalty. Fixed acceptance policy precedes calibration and test. No refit, test selection, or scale remapping.",
        "limitations": [
            "Predictions estimate this source's mean human frontal-photo rating; they do not measure objective worth or universal looks.",
            "102 selected London adults are a small convenience cohort, not representative population norms.",
            "Geometry omits appearance texture and labelled profiles. No four-pillar labels or strict 1–10 target exist in this cohort.",
            "All labels use the same source rater panel. This evaluates held-out people, not new rater populations.",
            "Split-conformal coverage is marginal for accepted exchangeable people from this source. It is not a per-person probability, individual-rater interval or guarantee on new cameras/populations.",
            "Simple standardized-feature range checks do not establish robust out-of-distribution detection.",
            "Selective diagnostic thresholds are reporting only and must not be chosen from test performance.",
            "Protected attributes are excluded as direct inputs; geometry can still correlate with them. This pilot has no fairness validation.",
            "Commercial-product permission and deployment readiness are not established. This artifact remains researchOnly.",
        ],
    }
    return model, report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("features", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--coverage", type=float, default=.9)
    args = parser.parse_args()
    records = [json.loads(line) for line in args.features.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    model, report = train(records, args.seed, args.coverage)
    report["featureFileSha256"] = hashlib.sha256(args.features.read_bytes()).hexdigest()
    args.out.mkdir(parents=True, exist_ok=True)
    for name, value in (("research-model.json", model), ("report.json", report)):
        (args.out / name).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"model": model["evaluation"], "calibration": model["calibration"], "bootstrap": report["bootstrap"], "output": str(args.out)}, allow_nan=False))
