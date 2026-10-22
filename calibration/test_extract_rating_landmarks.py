"""Local extraction contract tests; no human identities, photos, or ratings.

Metadata below is synthetic input for validation tests, not training data.
Vision inference is mocked; a separate manual smoke checks the pinned model.
"""
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import numpy as np
from PIL import Image

from extract_rating_landmarks import (
    extract_one,
    prepare_photo,
    validate_manifest,
    validate_paths,
    verify_model,
)


def fixture():
    return {"id": "synthetic-photo", "subject_id": "synthetic-person", "identity_verified": True,
            "view": "front", "image_path": "test-photo.png",
            "rating": {"mean": 3, "count": 5, "scale": [1, 5]}}


class ManifestTests(unittest.TestCase):
    def test_local_paths_resolve_against_manifest_directory(self):
        rows = validate_manifest([fixture()], Path("/tmp/study/manifest.jsonl"))
        self.assertEqual(rows[0]["image_path"], "/tmp/study/test-photo.png")

    def test_labels_may_be_joined_in_the_next_stage_without_inventing_them(self):
        row = fixture()
        del row["rating"]
        validated = validate_manifest([row], Path("/tmp/study/manifest.jsonl"))
        self.assertIsNone(validated[0]["rating"])

    def test_person_identity_must_be_declared_not_inferred(self):
        row = fixture()
        del row["subject_id"]
        with self.assertRaises(ValueError):
            validate_manifest([row], Path("/tmp/manifest.jsonl"))
        row = fixture()
        row["identity_verified"] = False
        with self.assertRaises(ValueError):
            validate_manifest([row], Path("/tmp/manifest.jsonl"))

    def test_two_photos_of_one_subject_fail(self):
        first = fixture()
        second = {**fixture(), "id": "second-photo"}
        with self.assertRaises(ValueError):
            validate_manifest([first, second], Path("/tmp/manifest.jsonl"))

    def test_remote_image_url_is_not_a_local_path(self):
        for remote in ["https://example.invalid/photo.jpg", "data:image/png;base64,AA==", "file:///tmp/photo.jpg"]:
            with self.subTest(remote=remote), self.assertRaises(ValueError):
                validate_manifest([{**fixture(), "image_path": remote}], Path("/tmp/manifest.jsonl"))

    def test_invalid_or_mixed_rating_scales_fail(self):
        for rating in [{"mean": float("nan"), "count": 5, "scale": [1, 5]},
                       {"mean": 3, "count": 1, "scale": [1, 5]},
                       {"mean": 3, "count": 5, "scale": [5, 1]}]:
            with self.subTest(rating=rating), self.assertRaises(ValueError):
                validate_manifest([{**fixture(), "rating": rating}], Path("/tmp/manifest.jsonl"))
        second = {**fixture(), "id": "other-photo", "subject_id": "other-person",
                  "rating": {"mean": 6, "count": 5, "scale": [0, 10]}}
        with self.assertRaises(ValueError):
            validate_manifest([fixture(), second], Path("/tmp/manifest.jsonl"))

    def test_outputs_cannot_replace_inputs_or_existing_outputs(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "manifest.jsonl"
            manifest.write_text("", encoding="utf8")
            rows = validate_manifest([fixture()], manifest)
            with self.assertRaises(ValueError):
                validate_paths(manifest, manifest, root / "pixels", root / "model.task", rows)
            output = root / "existing.jsonl"
            output.write_text("preserve", encoding="utf8")
            with self.assertRaises(ValueError):
                validate_paths(manifest, output, root / "pixels", root / "model.task", rows)
            self.assertEqual(output.read_text(encoding="utf8"), "preserve")

    def test_unrecognized_model_bytes_fail_integrity_check(self):
        with TemporaryDirectory() as directory:
            model = Path(directory) / "model.task"
            model.write_bytes(b"not the pinned model")
            with self.assertRaises(ValueError):
                verify_model(model)


class PhotoPreparationTests(unittest.TestCase):
    def test_exif_orientation_precedes_resize_and_rgb_conversion(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "rotated.jpg"
            photo = Image.new("RGB", (1200, 800), (128, 128, 128))
            exif = Image.Exif()
            exif[274] = 6  # Rotate 90 degrees clockwise.
            photo.save(path, exif=exif)
            prepared, metadata = prepare_photo(path, 768)
            self.assertEqual(prepared.size, (512, 768))
            self.assertEqual(prepared.mode, "RGB")
            self.assertEqual(metadata["sourceSize"], [1200, 800])

    def test_unreadable_photo_row_is_retained_with_no_invented_landmarks(self):
        row = {**fixture(), "image_path": "/tmp/facelab-definitely-absent-fixture.png"}
        output = extract_one(row, None, None, None, Path("/tmp"), 2048)
        self.assertEqual(output["id"], row["id"])
        self.assertEqual(output["subject_id"], row["subject_id"])
        self.assertEqual(output["detection_issue"], "image-error")
        self.assertEqual(output["landmarks"], [])
        self.assertIsNone(output["rgba_path"])

    def test_no_face_and_multiple_faces_are_retained_as_failures(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "flat.png"
            Image.new("RGB", (64, 64), (128, 128, 128)).save(path)
            row = {**fixture(), "image_path": str(path)}
            mp = SimpleNamespace(Image=lambda **kwargs: kwargs, ImageFormat=SimpleNamespace(SRGB="RGB"))
            for count, issue in [(0, "no-face"), (2, "multiple-faces")]:
                result = SimpleNamespace(face_landmarks=[[] for _ in range(count)])
                model = SimpleNamespace(detect=lambda _, result=result: result)
                output = extract_one(row, model, mp, np, Path(directory), 2048)
                self.assertEqual(output["detection_issue"], issue)
                self.assertEqual(output["detection_faces"], count)
                self.assertEqual(output["landmarks"], [])
                self.assertIsNone(output["rgba_path"])


if __name__ == "__main__":
    unittest.main()
