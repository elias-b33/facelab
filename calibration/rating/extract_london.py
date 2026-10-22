"""Extract automatic frontal landmarks and original human labels locally.

No downloads, model training, identity recognition, profile-label transfer, or
1--10 conversion. The source images remain outside the repository. MediaPipe's
official Python task uses the same model file as the browser application.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image

SOURCE_URL = "https://figshare.com/articles/dataset/Face_Research_Lab_London_Set/5047666"
EXPECTED_MODEL_SHA256 = "64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_model(path: Path) -> str:
    digest = sha256(path)
    if digest != EXPECTED_MODEL_SHA256:
        raise ValueError("Detector SHA-256 does not match the application's pinned face_landmarker.task")
    return digest


def detection_failure_record(record: dict, detected_faces: int) -> dict:
    """Keep failed subjects in the denominator without fabricating measurements."""
    return {**record, "scanInput": None,
            "detectionFailure": {"reason": "expected_exactly_one_face", "detectedFaces": detected_faces}}


def load_labels(root: Path) -> tuple[list[dict], dict]:
    """Load human ratings, preserving their documented photograph/scale scope."""
    info_path = root / "archives/london_faces_info.csv"
    rating_path = root / "archives/london_faces_ratings.csv"
    metadata_path = root / "source_metadata.json"
    source_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if source_metadata.get("license", {}).get("name") != "CC BY 4.0":
        raise ValueError("Expected original London source metadata and licence evidence")
    with info_path.open(encoding="utf-8-sig", newline="") as stream:
        people = list(csv.DictReader(stream))
    with rating_path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        ratings = {"X" + p["face_id"]: [] for p in people}
        if not set(ratings).issubset(reader.fieldnames or []):
            raise ValueError("Source person IDs do not match original human rating columns")
        for row in reader:
            for key, values in ratings.items():
                if row[key] not in ("", "NA", "NaN"):
                    value = float(row[key])
                    if not np.isfinite(value) or not 1 <= value <= 7:
                        raise ValueError("Human rating outside documented 1--7 scale")
                    values.append(value)
    source = {
        "url": SOURCE_URL,
        "license": "CC BY 4.0",
        "attribution": "DeBruine, Lisa; Jones, Benedict (2017). Face Research Lab London Set. doi:10.6084/m9.figshare.5047666",
        "commercialProductPermissionVerified": False,
        "consentScope": "Published signed consent covers lab/web studies and research illustration; specific commercial product use is not established here.",
        "metadataSha256": sha256(metadata_path),
        "humanRatingsSha256": sha256(rating_path),
        "personInfoSha256": sha256(info_path),
    }
    records = []
    for person in sorted(people, key=lambda p: p["face_id"]):
        subject = person["face_id"].zfill(3)
        age = person.get("face_age")
        if age not in (None, "", "NA") and float(age) < 18:
            raise ValueError("This research pilot requires adult subjects")
        values = np.asarray(ratings["X" + person["face_id"]], dtype=float)
        if len(values) < 3:
            raise ValueError("A subject is missing independent human ratings")
        records.append({
            "id": f"london-{subject}-front",
            "subjectId": f"london-{subject}",
            "dataset": "london",
            "source": source,
            "label": {"mean": float(values.mean()), "sd": float(values.std(ddof=1)),
                      "count": len(values), "min": 1, "max": 7,
                      "scope": "neutral_front_only", "kind": "mean-human-photo-rating"},
            "imageRelativePath": f"images/neutral_front/neutral_front/{subject}_03.jpg",
        })
    if len({r["subjectId"] for r in records}) != len(records):
        raise ValueError("Duplicate source person IDs")
    return records, source


def extract(root: Path, model_path: Path, output: Path) -> dict:
    detector_hash = verify_model(model_path)
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision

    records, source = load_labels(root)
    options = vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
        running_mode=vision.RunningMode.IMAGE,
        num_faces=2,  # A second face must reject the photograph, not be hidden.
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=True,
        min_face_detection_confidence=.5,
        min_face_presence_confidence=.5,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    rgba_directory = output.parent / "rgba"
    rgba_directory.mkdir(exist_ok=True)
    successes, failures = 0, []
    with vision.FaceLandmarker.create_from_options(options) as detector, output.open("w", encoding="utf-8") as stream:
        for record in records:
            path = root / record.pop("imageRelativePath")
            with Image.open(path) as image:
                rgb = np.asarray(image.convert("RGB"))
                rgba = image.convert("RGBA").tobytes()
                width, height = image.size
            record["detector"] = {"name": "MediaPipe Face Landmarker", "version": mp.__version__,
                                  "modelSha256": detector_hash, "imageSha256": sha256(path),
                                  "imageQualityPixelsIncluded": False}
            result = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
            if len(result.face_landmarks) != 1:
                record = detection_failure_record(record, len(result.face_landmarks))
                failures.append({"id": record["id"], "reason": "expected_exactly_one_face",
                                 "detectedFaces": len(result.face_landmarks)})
                stream.write(json.dumps(record, allow_nan=False) + "\n")
                continue
            matrix = result.facial_transformation_matrixes[0]
            rgba_path = rgba_directory / (record["id"] + ".rgba")
            rgba_path.write_bytes(rgba)
            record["rgbaPath"] = str(rgba_path.resolve())
            record["detector"]["imageQualityPixelsIncluded"] = True
            record["scanInput"] = {
                "landmarks": [{"x": float(p.x), "y": float(p.y), "z": float(p.z)} for p in result.face_landmarks[0]],
                "imageWidth": width, "imageHeight": height,
                "transformationMatrix": [float(matrix[r][c]) for c in range(4) for r in range(4)],
                "blendshapes": {b.category_name: float(b.score) for b in result.face_blendshapes[0]},
            }
            stream.write(json.dumps(record, allow_nan=False) + "\n")
            successes += 1
            if successes % 20 == 0:
                print(f"Extracted {successes}/{len(records)} frontal photographs", flush=True)
    report = {"source": source, "inputSubjects": len(records), "extractedSubjects": successes,
              "failedSubjects": failures, "detectorModelSha256": detector_hash,
              "mediapipeVersion": mp.__version__, "labelScale": [1, 7],
              "limitations": ["Landmarks are automatically estimated, not manual ground truth.",
                              "Raw RGBA sidecars are supplied for the application's pixel quality gates; those heuristic gates are not capture certification.",
                              "Labels describe neutral frontal photographs only. Profiles are not labelled."]}
    output.with_suffix(".extraction.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path(os.environ.get("LOCALAPPDATA", ".")) / "FaceIQ/datasets/london")
    parser.add_argument("--model", type=Path, required=True, help="The app's pinned face_landmarker.task file")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = extract(args.data_root, args.model, args.out)
    print(json.dumps({"subjects": report["extractedSubjects"], "failures": len(report["failedSubjects"]), "output": str(args.out)}))
