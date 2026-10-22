"""Tests of the training protocol using explicitly synthetic numeric fixtures.

These fixtures validate software behavior; none measures real-face accuracy.
Run: python -m unittest discover -s calibration -p 'test_*.py'
"""
import copy
import math
import unittest

import numpy as np

from train_rating import (
    CONTRACT,
    FEATURE_KEYS,
    FEATURE_SCHEMA,
    conformal_quantile,
    predict_artifact,
    split_subjects,
    train_model,
)


def synthetic_corpus(count=240):
    generator = np.random.default_rng(20261022)
    measurements = generator.normal(size=(count, len(FEATURE_KEYS)))
    labels = np.clip(3 + .45 * measurements[:, 0] - .25 * measurements[:, 1]
                     + .08 * measurements[:, 2] ** 2, 1, 5)
    rows = [{
        "id": f"synthetic-photo-{index:04d}",
        "subject_id": f"synthetic-person-{index:04d}",
        "identity_verified": True,
        "view": "front",
        "feature_schema": FEATURE_SCHEMA,
        "engine_version": "0.2.0-rating",
        "quality_pass": True,
        "features": dict(zip(FEATURE_KEYS, map(float, vector))),
        "confidence": dict.fromkeys(FEATURE_KEYS, 1),
        "rating": {"mean": float(labels[index]), "count": 10, "scale": [1, 5]},
    } for index, vector in enumerate(measurements)]
    manifest = {
        "datasetId": "synthetic-software-fixture",
        "datasetName": "Generated numbers for software verification only",
        "sourceUrl": "https://example.invalid/facelab-synthetic-test",
        "license": "Synthetic test data; not human face ratings",
        "usage": "research",
        "identityProtocol": "Each generated row has its own synthetic identifier; no real people.",
        "synthetic": True,
        "featureSchema": FEATURE_SCHEMA,
        "engineVersion": "0.2.0-rating",
        "target": {"label": "Synthetic test target", "min": 1, "max": 5,
                   "population": "Generated numbers only",
                   "protocol": "Deterministic test function; no human raters."},
    }
    return rows, manifest


class TrainingProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows, cls.manifest = synthetic_corpus()
        cls.artifact, cls.report, cls.splits = train_model(cls.rows, cls.manifest)
        cls.by_subject = {row["subject_id"]: row for row in cls.rows}

    def test_partitions_are_disjoint_complete_and_independent_of_input_order(self):
        split_sets = [set(part) for part in self.splits.values()]
        self.assertEqual(set.union(*split_sets), set(self.by_subject))
        self.assertEqual(sum(map(len, split_sets)), len(self.rows))
        for index, left in enumerate(split_sets):
            for right in split_sets[index + 1:]:
                self.assertTrue(left.isdisjoint(right))
        self.assertEqual(self.splits, split_subjects(list(reversed(self.rows))))

    def test_repeated_training_has_stable_parameters_and_model_identity(self):
        repeated, report, assignments = train_model(list(reversed(self.rows)), self.manifest)
        self.assertEqual(assignments, self.splits)
        self.assertEqual(report, self.report)
        self.assertEqual({key: value for key, value in repeated.items() if key != "createdAt"},
                         {key: value for key, value in self.artifact.items() if key != "createdAt"})

    def test_standardization_uses_only_training_people(self):
        training = [self.by_subject[subject] for subject in self.splits["train"]]
        for feature in self.artifact["features"]:
            values = np.asarray([row["features"][feature["key"]] for row in training])
            self.assertAlmostEqual(feature["mean"], float(values.mean()), places=12)
            self.assertAlmostEqual(feature["scale"], float(values.std()), places=12)
            self.assertEqual(feature["min"], float(values.min()))
            self.assertEqual(feature["max"], float(values.max()))

    def test_test_labels_cannot_change_weights_selection_or_calibration(self):
        changed = copy.deepcopy(self.rows)
        test_subjects = set(self.splits["test"])
        for row in changed:
            if row["subject_id"] in test_subjects:
                row["rating"]["mean"] = 6 - row["rating"]["mean"]
        artifact, report, _ = train_model(changed, self.manifest)
        self.assertEqual(artifact["features"], self.artifact["features"])
        self.assertEqual(artifact["intercept"], self.artifact["intercept"])
        self.assertEqual(artifact["calibration"], self.artifact["calibration"])
        self.assertEqual(report["selected"], self.report["selected"])
        self.assertNotEqual(artifact["evaluation"]["mae"], self.artifact["evaluation"]["mae"])

    def test_calibration_labels_cannot_change_weights_or_hyperparameters(self):
        changed = copy.deepcopy(self.rows)
        calibration_subjects = set(self.splits["calibration"])
        for row in changed:
            if row["subject_id"] in calibration_subjects:
                row["rating"]["mean"] = 6 - row["rating"]["mean"]
        artifact, report, _ = train_model(changed, self.manifest)
        self.assertEqual(artifact["features"], self.artifact["features"])
        self.assertEqual(artifact["intercept"], self.artifact["intercept"])
        self.assertEqual(report["selected"], self.report["selected"])
        self.assertNotEqual(artifact["calibration"]["absoluteErrorQuantile"],
                            self.artifact["calibration"]["absoluteErrorQuantile"])

    def test_mean_baseline_is_fitted_on_training_labels(self):
        training_mean = np.mean([self.by_subject[subject]["rating"]["mean"]
                                 for subject in self.splits["train"]])
        expected_mae = np.mean([abs(self.by_subject[subject]["rating"]["mean"] - training_mean)
                                for subject in self.splits["test"]])
        self.assertAlmostEqual(self.artifact["evaluation"]["baselineMae"], expected_mae, places=12)

    def test_source_scale_and_explicit_synthetic_provenance_are_preserved(self):
        self.assertEqual(self.report["sourceScale"], [1, 5])
        self.assertEqual((self.artifact["target"]["min"], self.artifact["target"]["max"]), (1, 5))
        self.assertIs(self.artifact["provenance"]["synthetic"], True)
        self.assertTrue(any("SYNTHETIC" in limitation for limitation in self.report["limitations"]))
        for subject in self.splits["test"]:
            prediction = predict_artifact(self.artifact, self.by_subject[subject]["features"])
            self.assertIs(prediction["ok"], True)
            self.assertLessEqual(1, prediction["lower"])
            self.assertLessEqual(prediction["lower"], prediction["value"])
            self.assertLessEqual(prediction["value"], prediction["upper"])
            self.assertLessEqual(prediction["upper"], 5)

    def test_abstentions_stay_in_the_coverage_denominator(self):
        changed = copy.deepcopy(self.rows)
        by_subject = {row["subject_id"]: row for row in changed}
        selected = self.splits["test"][:4]
        by_subject[selected[0]]["features"][FEATURE_KEYS[0]] = None
        by_subject[selected[1]]["confidence"][FEATURE_KEYS[0]] = .1
        by_subject[selected[2]]["quality_pass"] = False
        by_subject[selected[3]]["features"][FEATURE_KEYS[0]] = 100
        artifact, report, assignments = train_model(changed, self.manifest)
        audit = report["splitAudit"]["test"]
        self.assertEqual(audit["unusableRows"], 4)
        self.assertEqual(audit["totalRows"], len(assignments["test"]))
        self.assertEqual(audit["acceptedRows"], len(assignments["test"]) - 4)
        self.assertEqual(set(audit["reasons"]),
                         {"missing-feature", "low-feature-confidence", "photo-quality", "out-of-domain"})
        self.assertEqual(artifact["evaluation"]["testUnusableRows"], 4)
        self.assertAlmostEqual(artifact["evaluation"]["testCoverage"], audit["acceptedRows"] / audit["totalRows"])
        self.assertEqual(artifact["features"], self.artifact["features"])

    def test_constant_training_features_are_finite_and_inert(self):
        changed = copy.deepcopy(self.rows)
        for row in changed:
            row["features"][FEATURE_KEYS[-1]] = 3
        artifact, _, _ = train_model(changed, self.manifest)
        feature = artifact["features"][-1]
        self.assertEqual(feature["mean"], 3)
        self.assertEqual(feature["scale"], 1)
        self.assertAlmostEqual(feature["linear"], 0, places=12)
        self.assertAlmostEqual(feature["quadratic"], 0, places=12)
        self.assertTrue(math.isfinite(artifact["calibration"]["absoluteErrorQuantile"]))

    def test_duplicate_people_or_unverified_identities_are_rejected(self):
        for field, value in (("subject_id", self.rows[1]["subject_id"]), ("identity_verified", False)):
            with self.subTest(field=field):
                changed = copy.deepcopy(self.rows)
                changed[0][field] = value
                with self.assertRaises(ValueError):
                    train_model(changed, self.manifest)

    def test_nonfinite_or_boolean_numbers_are_rejected(self):
        for field, key in (("features", FEATURE_KEYS[0]), ("confidence", FEATURE_KEYS[0]), ("rating", "mean")):
            for invalid in (float("nan"), float("inf"), float("-inf"), True):
                with self.subTest(field=field, invalid=invalid):
                    changed = copy.deepcopy(self.rows)
                    changed[0][field][key] = invalid
                    with self.assertRaises(ValueError):
                        train_model(changed, self.manifest)

    def test_mixed_source_scales_and_insufficient_raters_are_rejected(self):
        for field, value in (("scale", [0, 10]), ("count", CONTRACT["minimumRatersPerPhoto"] - 1)):
            with self.subTest(field=field):
                changed = copy.deepcopy(self.rows)
                changed[0]["rating"][field] = value
                with self.assertRaises(ValueError):
                    train_model(changed, self.manifest)

    def test_insufficient_people_are_not_relabelled_as_a_valid_holdout(self):
        with self.assertRaises(ValueError):
            train_model(self.rows[:99], self.manifest)


class ConformalIntervalTests(unittest.TestCase):
    def test_finite_sample_order_statistic_is_used_without_interpolation(self):
        self.assertEqual(conformal_quantile(np.arange(20, dtype=float) / 10, .9), 1.8)

    def test_infeasible_coverage_or_invalid_residuals_are_rejected(self):
        for coverage in (0, 1, .99, .999, float("nan")):
            with self.subTest(coverage=coverage), self.assertRaises(ValueError):
                conformal_quantile(np.arange(20, dtype=float), coverage)
        for residuals in (np.arange(19), np.full(20, -1), np.full(20, np.nan)):
            with self.assertRaises(ValueError):
                conformal_quantile(residuals, .9)


if __name__ == "__main__":
    unittest.main()
