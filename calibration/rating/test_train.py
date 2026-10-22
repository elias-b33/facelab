"""Numerical protocol tests. Artificial fixtures below are never accuracy data."""
import copy
from pathlib import Path
import tempfile
import unittest

import numpy as np

from train import (FEATURE_SCHEMA, POLICY, assign_subject_splits, conformal_radius,
                   fit_transform, metrics, policy_check, train, usable, validate_records)
from extract_london import detection_failure_record, verify_model


def fixtures(n=120):
    # These are artificial regression numbers, not faces or human annotations.
    rows = []
    for i in range(n):
        value = (i % 17 - 8) / 8
        other = (i % 11 - 5) / 5
        rows.append({
            "id": f"artificial-capture-{i}", "subjectId": f"artificial-person-{i}", "dataset": "unit-test-only",
            "source": {"url": "https://example.invalid/unit-test", "license": "Artificial numeric fixture",
                       "humanRatingsSha256": "artificial-placeholder-not-human-evidence"},
            "label": {"kind": "mean-human-photo-rating", "scope": "neutral_front_only", "min": 1, "max": 7,
                      "mean": 4 + .8 * value - .1 * other, "count": 10},
            "features": {"schemaVersion": FEATURE_SCHEMA,
                         "values": {"legacy.a": value, "legacy.b": other, "harmony.unobserved": None},
                         "quality": {"frontUsable": True, "blockingReasons": []}},
        })
    return rows


class ProtocolTests(unittest.TestCase):
    def test_detector_failure_preserves_identity_label_and_missing_measurements(self):
        original = fixtures()[0]
        original["detector"] = {"modelSha256": "test-hash", "imageSha256": "test-image-hash"}
        for detected in (0, 2):
            failed = detection_failure_record(original, detected)
            self.assertEqual(failed["subjectId"], original["subjectId"])
            self.assertEqual(failed["label"], original["label"])
            self.assertEqual(failed["detector"], original["detector"])
            self.assertIsNone(failed["scanInput"])
            self.assertEqual(failed["detectionFailure"]["detectedFaces"], detected)

    def test_wrong_detector_hash_is_rejected_before_extraction(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "different.task"
            path.write_bytes(b"not-the-pinned-detector")
            with self.assertRaisesRegex(ValueError, "pinned"):
                verify_model(path)

    def test_group_split_keeps_repeated_captures_and_input_order_independent(self):
        rows = fixtures()
        repeated = copy.deepcopy(rows[0])
        repeated["id"] = "second-view-same-person"
        rows.append(repeated)
        first = assign_subject_splits(rows, 123)
        self.assertEqual(first, assign_subject_splits(list(reversed(rows)), 123))
        self.assertEqual(len(first), 120)
        self.assertEqual(set(first.values()), {"fit", "tuning", "calibration", "test"})

    def test_reject_missing_identity_and_mixed_target_scale(self):
        rows = fixtures()
        del rows[0]["subjectId"]
        with self.assertRaisesRegex(ValueError, "subjectId"):
            validate_records(rows)
        rows = fixtures()
        rows[0]["label"]["max"] = 10
        with self.assertRaisesRegex(ValueError, "source scale"):
            validate_records(rows)

    def test_source_photo_scope_cannot_be_relabelled_as_profile(self):
        rows = fixtures()
        rows[0]["label"]["scope"] = "profile"
        with self.assertRaisesRegex(ValueError, "frontal-photo"):
            validate_records(rows)

    def test_unobserved_features_are_excluded_and_missingness_is_explicit(self):
        features, excluded = fit_transform(fixtures())
        self.assertEqual(excluded, ["harmony.unobserved"])
        self.assertEqual([f["key"] for f in features], ["legacy.a", "legacy.b"])
        missing = copy.deepcopy(fixtures()[0])
        missing["features"]["values"]["legacy.a"] = None
        self.assertIn("insufficient_feature_coverage", policy_check(missing, features, POLICY)[1])

    def test_out_of_range_and_capture_failure_abstain(self):
        rows = fixtures()
        features, _ = fit_transform(rows)
        unusual = copy.deepcopy(rows[0])
        unusual["features"]["values"]["legacy.a"] = 1000
        self.assertFalse(policy_check(unusual, features, POLICY)[0])
        unusual = copy.deepcopy(rows[0])
        unusual["features"]["quality"]["frontUsable"] = False
        self.assertIn("capture_quality", policy_check(unusual, features, POLICY)[1])

    def test_capture_gate_matches_browser_and_never_coerces_features(self):
        row = fixtures()[0]
        self.assertTrue(usable(row))
        row["features"]["quality"]["blockingReasons"] = ["blocked despite true flag"]
        self.assertFalse(usable(row))
        row["features"]["quality"]["blockingReasons"] = ""
        self.assertFalse(usable(row))
        row["features"]["quality"]["blockingReasons"] = []
        row["features"]["schemaVersion"] = "other-schema"
        self.assertFalse(usable(row))
        for invalid in (True, "1.0", float("inf"), float("nan")):
            rows = fixtures()
            rows[0]["features"]["values"]["legacy.a"] = invalid
            with self.assertRaisesRegex(ValueError, "finite observed"):
                validate_records(rows)

    def test_finite_sample_conformal_order_and_insufficient_sample(self):
        self.assertEqual(conformal_radius(list(range(1, 10)), .9), 9)
        with self.assertRaisesRegex(ValueError, "Too few"):
            conformal_radius(list(range(1, 9)), .9)
        self.assertEqual(conformal_radius([.1] * 19 + [9.0], .9), .1)

    def test_test_labels_and_features_cannot_change_fit_or_calibration(self):
        rows = fixtures()
        assignment = assign_subject_splits(rows, 123)
        original, _ = train(rows, seed=123)
        edited = copy.deepcopy(rows)
        for row in edited:
            if assignment[row["subjectId"]] == "test":
                row["label"]["mean"] = 2
                row["features"]["values"]["legacy.a"] *= .1
        after, _ = train(edited, seed=123)
        for field in ("features", "intercept", "calibration", "policy", "training"):
            self.assertEqual(original[field], after[field])
        self.assertNotEqual(original["evaluation"]["mae"], after["evaluation"]["mae"])

    def test_calibration_labels_cannot_change_coefficients(self):
        rows = fixtures()
        assignment = assign_subject_splits(rows, 123)
        original, _ = train(rows, seed=123)
        for row in rows:
            if assignment[row["subjectId"]] == "calibration":
                row["label"]["mean"] = 6.5
        after, _ = train(rows, seed=123)
        self.assertEqual(original["features"], after["features"])
        self.assertEqual(original["intercept"], after["intercept"])
        self.assertNotEqual(original["calibration"]["radius"], after["calibration"]["radius"])

    def test_no_silent_calibration_fallback_after_abstention(self):
        rows = fixtures()
        assignment = assign_subject_splits(rows, 123)
        for row in rows:
            if assignment[row["subjectId"]] == "calibration":
                row["features"]["values"]["legacy.a"] = 1000
        with self.assertRaisesRegex(ValueError, "calibration requires|Too few"):
            train(rows, seed=123)

    def test_constant_predictions_have_no_rank_correlation(self):
        self.assertIsNone(metrics(np.asarray([1, 2, 3]), np.asarray([2, 2, 2]))["spearman"])

    def test_target_name_and_report_provenance_follow_actual_inputs(self):
        rows = fixtures()
        for row in rows:
            row["label"]["max"] = 5
            row["engineVersion"] = "unit-test-engine"
            row["detector"] = {"name": "numeric fixture", "version": "test", "imageSha256": row["id"]}
            row["pixelGatesEvaluated"] = True
        model, report = train(rows, seed=123)
        self.assertIn("1–5", model["target"]["name"])
        self.assertEqual(report["engineVersions"], ["unit-test-engine"])
        self.assertEqual(report["captureProvenance"][0]["detector"], rows[0]["detector"])
        self.assertTrue(report["captureProvenance"][0]["pixelGatesEvaluated"])


if __name__ == "__main__":
    unittest.main()
