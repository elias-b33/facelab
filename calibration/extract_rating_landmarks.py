#!/usr/bin/env python3
"""Extract local, preselected frontal photos for the shared rating pipeline.

No photos, labels, models, or datasets are downloaded. Each manifest JSONL row
must contain id, subject_id, identity_verified, view, and image_path. Optional
rating {mean, count, scale} can instead be joined by exact ID in the next stage.
Identity and label declarations come from the curator;
they are never inferred from a filename or from the face.

The output preserves every manifest row, including unreadable images, no-face,
and multi-face failures. Successful detections include full-precision landmarks,
blendshapes, pose, and a local raw RGBA file for the production engine's image
quality checks. Run export-rating-features.ts next, then train_rating.py.

The task model bytes match the browser's pinned model. Python CPU inference,
Pillow resizing, browser inference, and stored JPEG reanalysis can differ;
cross-backend parity and repeat-photo stability still need measurement.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
from importlib.metadata import version
import json
import math
from pathlib import Path
import re
from typing import Any
import warnings


MODEL_SHA256 = "64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff"
DEFAULT_MODEL = Path(__file__).resolve().parents[1] / "apps/web/public/mediapipe/face_landmarker.task"
DEFAULT_MAX_LONG_EDGE = 2048
MAX_SOURCE_PIXELS = 40_000_000
MAX_MANIFEST_ROWS = 100_000


def _text(value: Any, field: str, maximum: int = 4000) -> str:
    if not isinstance(value, str) or not value or value != value.strip() or len(value) > maximum:
        raise ValueError(f"{field} must be a nonempty trimmed string of at most {maximum} characters")
    return value


def _number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{field} must be a finite number")
    return float(value)


def _strict_json(text: str) -> Any:
    def invalid_constant(value: str) -> None:
        raise ValueError(f"Non-finite JSON constant {value} is not permitted")
    return json.loads(text, parse_constant=invalid_constant)


def validate_manifest(rows: Any, manifest_path: Path) -> list[dict[str, Any]]:
    """Validate metadata before importing vision dependencies or writing files.

    Missing/unreadable image files are handled per row during extraction so
    they remain in the evaluation denominator. Structural/identity ambiguity
    fails the whole manifest instead of silently discarding people.
    """
    if not isinstance(rows, list) or not 1 <= len(rows) <= MAX_MANIFEST_ROWS:
        raise ValueError(f"The photo manifest must contain 1–{MAX_MANIFEST_ROWS} JSONL rows")
    ids: set[str] = set()
    subjects: set[str] = set()
    validated: list[dict[str, Any]] = []
    source_scale: tuple[float, float] | None = None
    for number, row in enumerate(rows, 1):
        if not isinstance(row, dict):
            raise ValueError(f"Row {number} must be an object")
        photo_id = _text(row.get("id"), f"row {number} id")
        subject = _text(row.get("subject_id"), f"{photo_id} subject_id")
        if photo_id in ids or subject in subjects:
            raise ValueError(f"{photo_id}: duplicate photo/person; preselect one frontal photo per verified person")
        ids.add(photo_id)
        subjects.add(subject)
        if row.get("identity_verified") is not True:
            raise ValueError(f"{photo_id}: identity_verified must be true after a separate identity audit")
        if row.get("view") != "front":
            raise ValueError(f"{photo_id}: this extractor accepts frontal photos only")
        path_text = _text(row.get("image_path"), f"{photo_id} image_path", maximum=10000)
        # A Windows drive is a local path; URL/data schemes are not.
        if re.match(r"^[A-Za-z][A-Za-z0-9+.-]*:", path_text) and not re.match(r"^[A-Za-z]:[\\/]", path_text):
            raise ValueError(f"{photo_id}: image_path must be a local filesystem path, not a URL")
        image_path = Path(path_text).expanduser()
        if not image_path.is_absolute():
            image_path = manifest_path.resolve().parent / image_path
        rating = row.get("rating")
        if rating is not None:
            if not isinstance(rating, dict):
                raise ValueError(f"{photo_id}: supplied rating metadata must be an object")
            score = _number(rating.get("mean"), f"{photo_id} rating.mean")
            count = _number(rating.get("count"), f"{photo_id} rating.count")
            if count != int(count) or count < 5 or count > 2**53 - 1:
                raise ValueError(f"{photo_id}: rating.count must be an integer >= 5 independent panel ratings")
            scale = rating.get("scale")
            if not isinstance(scale, list) or len(scale) != 2:
                raise ValueError(f"{photo_id}: rating.scale must preserve [min, max] from the rating protocol")
            lo, hi = (_number(value, f"{photo_id} rating.scale") for value in scale)
            if not 0 <= lo < hi <= 10 or not lo <= score <= hi:
                raise ValueError(f"{photo_id}: rating.mean must fit the declared source scale within 0–10")
            if source_scale is not None and source_scale != (lo, hi):
                raise ValueError("Do not mix different rating scales in one dataset")
            source_scale = (lo, hi)
            rating = {"mean": score, "count": int(count), "scale": [lo, hi]}
        validated.append({
            "id": photo_id, "subject_id": subject, "identity_verified": True,
            "view": "front", "image_path": str(image_path.resolve()),
            "rating": rating,
        })
    return validated


def validate_paths(manifest_path: Path, output_path: Path, pixels_dir: Path,
                   model_path: Path, rows: list[dict[str, Any]]) -> None:
    """Fail before writing if outputs could replace source metadata or photos."""
    output = output_path.resolve()
    inputs = {manifest_path.resolve(), model_path.resolve(), *(Path(row["image_path"]).resolve() for row in rows)}
    if output in inputs:
        raise ValueError("The output must not replace the manifest, model, or any source photo")
    if output.exists():
        raise ValueError("The output already exists; choose a new output filename")
    if pixels_dir.resolve() == output or pixels_dir.resolve() in inputs:
        raise ValueError("The pixel directory must be a directory separate from source files and the output")
    if pixels_dir.exists() and not pixels_dir.is_dir():
        raise ValueError("The pixel destination is not a directory")


def verify_model(model_path: Path) -> None:
    if not model_path.is_file():
        raise ValueError("The pinned local face_landmarker.task is missing; prepare the web model assets first")
    with model_path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if digest != MODEL_SHA256:
        raise ValueError("Face model SHA-256 differs from the model pinned by the web app")


def prepare_photo(path: Path, max_long_edge: int) -> tuple[Any, dict[str, Any]]:
    # Lazy imports keep manifest/path validation testable without MediaPipe.
    from PIL import Image, ImageOps

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(path) as original:
                if original.width * original.height > MAX_SOURCE_PIXELS:
                    raise ValueError("The source photo exceeds the 40-megapixel extraction limit")
                if getattr(original, "n_frames", 1) != 1:
                    raise ValueError("Use a still photo, not an animated or multipage image")
                source_size = [original.width, original.height]
                photo = ImageOps.exif_transpose(original).convert("RGB")
                scale = min(1.0, max_long_edge / max(photo.width, photo.height))
                if scale < 1:
                    # Match JavaScript Math.round for positive dimensions. Pixel
                    # interpolation remains explicitly Pillow LANCZOS, not browser.
                    size = (max(1, math.floor(photo.width * scale + .5)),
                            max(1, math.floor(photo.height * scale + .5)))
                    photo = photo.resize(size, resample=Image.Resampling.LANCZOS)
                photo.load()
    except Image.DecompressionBombError as error:
        raise ValueError("The source photo exceeds the image decoder's size limit") from error
    return photo, {"sourceSize": source_size, "maxLongEdge": max_long_edge,
                   "imageWidth": photo.width, "imageHeight": photo.height,
                   "exifTransposed": True, "color": "RGB", "resize": "Pillow-LANCZOS",
                   "pillowVersion": version("Pillow"), "modelSha256": MODEL_SHA256}


def _base_row(row: dict[str, Any]) -> dict[str, Any]:
    return {key: row[key] for key in ("id", "subject_id", "identity_verified", "view", "rating")}


def extract_one(row: dict[str, Any], landmarker: Any, mp: Any, np: Any,
                pixels_dir: Path, max_long_edge: int) -> dict[str, Any]:
    output = {**_base_row(row), "w": 0, "h": 0, "landmarks": [], "matrix": None,
              "blendshapes": {}, "mirrored": False, "rgba_path": None,
              "detection_faces": None, "detection_issue": None}
    try:
        photo, preprocessing = prepare_photo(Path(row["image_path"]), max_long_edge)
        output.update(w=photo.width, h=photo.height, preprocessing=preprocessing)
    except (OSError, ValueError, Warning) as error:
        output.update(detection_issue="image-error", detection_message=f"Image decode failed ({type(error).__name__})")
        return output
    try:
        result = landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.asarray(photo)))
    except Exception as error:
        # Preserve failures instead of improving apparent dataset coverage by
        # silently removing difficult photos. Model initialization fails outside
        # this loop, so a missing dependency cannot turn the corpus into failures.
        output.update(detection_issue="inference-error", detection_message=f"Local inference failed ({type(error).__name__})")
        return output
    count = len(result.face_landmarks)
    output["detection_faces"] = count
    if count != 1:
        output["detection_issue"] = "no-face" if count == 0 else "multiple-faces"
        return output
    landmarks = [{"x": float(p.x), "y": float(p.y), "z": float(p.z)} for p in result.face_landmarks[0]]
    if len(landmarks) not in (468, 478) or any(not math.isfinite(v) for p in landmarks for v in p.values()):
        output["detection_issue"] = "invalid-detection"
        return output
    blendshapes = {c.category_name: float(c.score) for c in (result.face_blendshapes[0] if result.face_blendshapes else [])}
    if any(not isinstance(key, str) or not key or not math.isfinite(value) or not 0 <= value <= 1 for key, value in blendshapes.items()):
        output["detection_issue"] = "invalid-detection"
        return output
    matrix = None
    if result.facial_transformation_matrixes:
        raw_matrix = np.asarray(result.facial_transformation_matrixes[0], dtype=np.float64)
        if raw_matrix.shape == (4, 4) and np.isfinite(raw_matrix).all():
            # Explicit row-major storage; the shared engine accepts this layout.
            matrix = [float(value) for value in raw_matrix.flatten(order="C")]
        else:
            output["pose_issue"] = "invalid-pose-matrix"
    rgba = photo.convert("RGBA").tobytes()
    digest = hashlib.sha256(rgba).hexdigest()
    name_digest = hashlib.sha256(f"{photo.width}x{photo.height}:".encode() + rgba).hexdigest()
    pixel_path = pixels_dir.resolve() / f"{name_digest}.rgba"
    try:
        with pixel_path.open("xb") as stream:
            stream.write(rgba)
    except FileExistsError:
        # Reusing identical pixels is safe; never overwrite unrelated content.
        with pixel_path.open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != digest:
                raise ValueError("An existing pixel artifact does not match its expected content")
    output.update(landmarks=landmarks, matrix=matrix, blendshapes=blendshapes,
                  rgba_path=str(pixel_path), rgba_sha256=digest)
    output["preprocessing"]["mediapipeVersion"] = version("mediapipe")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path, help="Local photo + verified person + human-label JSONL")
    parser.add_argument("--out", required=True, type=Path, help="New raw-landmark JSONL destination; all input rows are retained")
    parser.add_argument("--pixels-dir", required=True, type=Path, help="Private raw RGBA directory for the shared image-quality gates")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="Existing pinned web Face Landmarker task")
    parser.add_argument("--max-long-edge", type=int, default=DEFAULT_MAX_LONG_EDGE,
                        help="Maximum photo long edge, default 2048 to match initial browser upload")
    args = parser.parse_args()
    if not 256 <= args.max_long_edge <= 4096:
        parser.error("--max-long-edge must be between 256 and 4096 pixels")
    try:
        rows = [_strict_json(line) for line in args.manifest.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
        rows = validate_manifest(rows, args.manifest)
        validate_paths(args.manifest, args.out, args.pixels_dir, args.model, rows)
        verify_model(args.model)
        # CPU makes the chosen execution backend explicit and avoids GPU setup.
        import mediapipe as mp
        import numpy as np
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        options = vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(args.model.resolve()), delegate=mp_python.BaseOptions.Delegate.CPU),
            running_mode=vision.RunningMode.IMAGE, num_faces=2,
            min_face_detection_confidence=.5, min_face_presence_confidence=.5,
            output_face_blendshapes=True, output_facial_transformation_matrixes=True,
        )
        counts: Counter[str] = Counter()
        with vision.FaceLandmarker.create_from_options(options) as landmarker:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.pixels_dir.mkdir(parents=True, exist_ok=True)
            with args.out.open("x", encoding="utf-8") as stream:
                for row in rows:
                    output = extract_one(row, landmarker, mp, np, args.pixels_dir, args.max_long_edge)
                    stream.write(json.dumps(output, ensure_ascii=False, allow_nan=False) + "\n")
                    counts[output["detection_issue"] or "detected"] += 1
        print(json.dumps({"inputRows": len(rows), "outputRows": sum(counts.values()),
                          "detections": dict(counts), "modelSha256": MODEL_SHA256,
                          "note": "Only local measurements; no human-rating model was trained."}, indent=2))
    except ImportError as error:
        parser.exit(2, f"Missing extraction dependency: {error}. Install calibration/requirements-extraction.txt in a virtual environment.\n")
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as error:
        parser.exit(2, f"Photo extraction failed: {error}\n")


if __name__ == "__main__":
    main()
