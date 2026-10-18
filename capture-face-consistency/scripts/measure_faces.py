#!/usr/bin/env python3
"""Measure face-landmark ratios with local Vision or MediaPipe backends.

The script does not identify people. It measures landmarks in user-selected images.
All distances are normalized to the detected face bounding box.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import platform
import statistics
import struct
import subprocess
import sys
from pathlib import Path

from mediapipe_backend import BackendError, MediaPipeBackend, sha256_file


def decode_points(region: dict | None) -> list[tuple[float, float]]:
    if not region or not region.get("data"):
        return []
    count = int(region["count"])
    raw = base64.b64decode(region["data"])
    expected = count * 16
    if len(raw) != expected:
        raise ValueError(f"Unexpected landmark byte count: {len(raw)} != {expected}")
    return [struct.unpack_from("<dd", raw, index * 16) for index in range(count)]


def center(points: list[tuple[float, float]]) -> tuple[float, float] | None:
    if not points:
        return None
    return (
        sum(point[0] for point in points) / len(points),
        sum(point[1] for point in points) / len(points),
    )


def extent(points: list[tuple[float, float]]) -> tuple[float, float] | None:
    if not points:
        return None
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    return max(xs) - min(xs), max(ys) - min(ys)


def distance(a: tuple[float, float] | None, b: tuple[float, float] | None) -> float | None:
    if a is None or b is None:
        return None
    return math.hypot(a[0] - b[0], a[1] - b[1])


def angle(a: tuple[float, float] | None, b: tuple[float, float] | None) -> float | None:
    if a is None or b is None:
        return None
    return math.degrees(math.atan2(b[1] - a[1], b[0] - a[0]))


def safe_ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or abs(denominator) < 1e-9:
        return None
    return numerator / denominator


def side_centers(points: list[tuple[float, float]]) -> tuple[tuple[float, float] | None, tuple[float, float] | None]:
    middle = center(points)
    if middle is None:
        return None, None
    return (
        center([point for point in points if point[0] <= middle[0]]),
        center([point for point in points if point[0] > middle[0]]),
    )


def face_metrics(face: dict) -> dict[str, float | None]:
    points = {name: decode_points(region) for name, region in face["landmarks"].items()}
    left_eye = points["left_eye"]
    right_eye = points["right_eye"]
    left_pupil = center(points["left_pupil"]) or center(left_eye)
    right_pupil = center(points["right_pupil"]) or center(right_eye)
    left_brow = center(points["left_eyebrow"])
    right_brow = center(points["right_eyebrow"])
    eye_separation = distance(left_pupil, right_pupil)
    left_eye_size = extent(left_eye)
    right_eye_size = extent(right_eye)
    nose_size = extent(points["nose"])
    mouth_size = extent(points["outer_lips"])
    contour_size = extent(points["face_contour"])
    mouth_sides = side_centers(points["outer_lips"])

    lower_jaw_width = None
    jaw_taper = None
    contour = points["face_contour"]
    if contour and contour_size and contour_size[1] > 0:
        min_y = min(point[1] for point in contour)
        cutoff = min_y + contour_size[1] * 0.38
        lower_points = [point for point in contour if point[1] <= cutoff]
        lower_extent = extent(lower_points)
        lower_jaw_width = lower_extent[0] if lower_extent else None
        jaw_taper = safe_ratio(lower_jaw_width, contour_size[0])

    mouth_width = mouth_size[0] if mouth_size else None
    mouth_height = mouth_size[1] if mouth_size else None
    nose_width = nose_size[0] if nose_size else None
    contour_width = contour_size[0] if contour_size else None
    contour_height = contour_size[1] if contour_size else None

    return {
        "yaw_degrees": math.degrees(face["yaw_radians"]) if face.get("yaw_radians") is not None else None,
        "roll_degrees": math.degrees(face["roll_radians"]) if face.get("roll_radians") is not None else None,
        "pitch_degrees": math.degrees(face["pitch_radians"]) if face.get("pitch_radians") is not None else None,
        "eye_separation": eye_separation,
        "left_eye_width": left_eye_size[0] if left_eye_size else None,
        "right_eye_width": right_eye_size[0] if right_eye_size else None,
        "left_eye_openness": safe_ratio(left_eye_size[1], left_eye_size[0]) if left_eye_size else None,
        "right_eye_openness": safe_ratio(right_eye_size[1], right_eye_size[0]) if right_eye_size else None,
        "eye_line_tilt_degrees": angle(left_pupil, right_pupil),
        "left_brow_eye_gap": distance(left_brow, left_pupil),
        "right_brow_eye_gap": distance(right_brow, right_pupil),
        "nose_width": nose_width,
        "mouth_width": mouth_width,
        "mouth_height": mouth_height,
        "mouth_aspect": safe_ratio(mouth_height, mouth_width),
        "mouth_corner_tilt_degrees": angle(mouth_sides[0], mouth_sides[1]),
        "contour_width": contour_width,
        "contour_height": contour_height,
        "contour_aspect": safe_ratio(contour_height, contour_width),
        "lower_jaw_width": lower_jaw_width,
        "jaw_taper": jaw_taper,
        "nose_to_eye_separation": safe_ratio(nose_width, eye_separation),
        "mouth_to_eye_separation": safe_ratio(mouth_width, eye_separation),
    }


def rounded(value: float | None) -> float | None:
    return round(value, 4) if value is not None and math.isfinite(value) else None


def mean_values(*values: float | None) -> float | None:
    present = [value for value in values if value is not None]
    return statistics.fmean(present) if present else None


def profile_metrics(metrics: dict[str, float | None]) -> dict[str, float | None]:
    """Map raw script metrics to the names used by the identity-profile schema."""
    return {
        "eye_center_distance_over_face_width": rounded(metrics.get("eye_separation")),
        "single_eye_width_over_face_width": rounded(
            mean_values(metrics.get("left_eye_width"), metrics.get("right_eye_width"))
        ),
        "eye_height_over_eye_width": rounded(
            mean_values(metrics.get("left_eye_openness"), metrics.get("right_eye_openness"))
        ),
        "brow_eye_center_distance_over_face_width": rounded(
            mean_values(metrics.get("left_brow_eye_gap"), metrics.get("right_brow_eye_gap"))
        ),
        "nose_width_over_face_width": rounded(metrics.get("nose_width")),
        "nose_width_over_eye_center_distance": rounded(metrics.get("nose_to_eye_separation")),
        "mouth_width_over_face_width": rounded(metrics.get("mouth_width")),
        "mouth_width_over_eye_center_distance": rounded(metrics.get("mouth_to_eye_separation")),
        "mouth_height_over_mouth_width": rounded(metrics.get("mouth_aspect")),
        "lower_jaw_width_over_contour_width": rounded(metrics.get("jaw_taper")),
        "vision_lower_face_contour_height_over_width": rounded(metrics.get("contour_aspect")),
    }


def measurement_quality(face_pixel_min: float) -> str:
    if face_pixel_min >= 160:
        return "high"
    if face_pixel_min >= 96:
        return "medium"
    return "low"


def summarize(rows: list[dict]) -> dict[str, dict[str, float]]:
    if not rows:
        return {}
    summary: dict[str, dict[str, float]] = {}
    for key in rows[0]["metrics"]:
        values = [row["metrics"][key] for row in rows if row["metrics"].get(key) is not None]
        if not values:
            continue
        summary[key] = {
            "median": rounded(statistics.median(values)),
            "min": rounded(min(values)),
            "max": rounded(max(values)),
            "n": len(values),
        }
    return summary


def source_sha256(path: Path) -> str:
    """Normalize checkout line endings so Windows CRLF keeps the same contract."""
    return hashlib.sha256(path.read_text(encoding="utf-8").encode("utf-8")).hexdigest()


def measurement_contract(metadata: dict) -> dict:
    """Version the definition, code and preprocessing; never pool model families."""
    sources = [Path(__file__), Path(__file__).with_name("mediapipe_backend.py")]
    if metadata["name"] == "vision":
        sources.append(Path(__file__).with_name("extract_face_landmarks.jxa"))
    preprocessing = {
        "code_sha256": {path.name: source_sha256(path) for path in sources},
        "coordinate_space": "face-box-unit-square-x-right-y-up-v1",
        "orientation": metadata["orientation"],
        "pillow_version": metadata.get("pillow_version"),
        "numpy_version": metadata.get("numpy_version"),
    }
    metadata = dict(metadata)
    metadata["preprocessing_hash"] = hashlib.sha256(
        json.dumps(preprocessing, sort_keys=True).encode("utf-8")
    ).hexdigest()
    signature = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode("utf-8")).hexdigest()
    return {"detector": metadata, "measurement_signature": signature, "preprocessing": preprocessing}


def extract_vision(paths: list[str], cpu_only: bool = False) -> tuple[list[dict], dict]:
    if sys.platform != "darwin":
        raise BackendError("Vision requires macOS and osascript. Use --backend mediapipe --model PATH on Windows/Linux.")
    script = Path(__file__).with_name("extract_face_landmarks.jxa")
    try:
        process = subprocess.run(
            ["osascript", "-l", "JavaScript", str(script), *(["--cpu-only"] if cpu_only else []), *paths],
            check=False, capture_output=True, text=True, timeout=120,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise BackendError(f"Cannot run local Vision: {error}") from error
    if process.returncode:
        raise BackendError(f"Vision failed: {process.stderr.strip()}")
    try:
        images = json.loads(process.stdout)
        if not isinstance(images, list):
            raise ValueError("expected an image array")
    except (json.JSONDecodeError, ValueError) as error:
        raise BackendError(f"Cannot parse Vision output: {error}") from error
    for image in images:
        try:
            image["image_sha256"] = sha256_file(Path(image["path"]))
        except OSError as error:
            image["error"] = str(error)
    return images, {
        "name": "vision", "version": f"macOS-{platform.mac_ver()[0]}-request-revision-3",
        "model_sha256": None, "landmark_schema": "VNFaceLandmarks2D-revision-3",
        "face_box_definition": "Vision detected face bounding box",
        "uses_cpu_only": cpu_only,
        "orientation": "legacy CIImage extent; EXIF behavior requires native verification",
    }


def load_reviews(path: str | None) -> dict[str, dict]:
    if path is None:
        return {}
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("schema_version") != "1.0" or not isinstance(data.get("images"), list):
        raise ValueError("Review file requires schema_version=1.0 and an images array")
    for field in ("subject_id", "era_id"):
        if not isinstance(data.get(field), str) or not data[field].strip():
            raise ValueError(f"Review file needs one {field}; do not pool people or eras")
    reviews = {}
    for item in data["images"]:
        if not isinstance(item, dict) or not isinstance(item.get("path"), str) or not item["path"]:
            raise ValueError("Each review needs an image path (relative to the working directory)")
        key = str(Path(item["path"]).resolve())
        if key in reviews:
            raise ValueError(f"Duplicate review path: {item['path']}")
        index = item.get("face_index")
        if type(index) is not int or index < 0:
            raise ValueError("Review face_index must be a nonnegative integer")
        for field in ("subject_confirmed", "canonical_approved"):
            if type(item.get(field)) is not bool:
                raise ValueError(f"Review {field} must be a JSON boolean")
        if not isinstance(item.get("reviewer"), str) or not item["reviewer"].strip():
            raise ValueError("Review reviewer must be recorded")
        if not isinstance(item.get("source_id"), str) or not item["source_id"].strip():
            raise ValueError("Review source_id must link the image to the source manifest")
        if not isinstance(item.get("face_box"), dict):
            raise ValueError("Review needs face_box from the measurement/candidate output")
        reviews[key] = {**item, "subject_id": data["subject_id"], "era_id": data["era_id"]}
    return reviews


def candidate_boxes(image: dict) -> list[dict]:
    candidates = []
    for index, face in enumerate(image.get("faces", [])):
        box = face["bounding_box"]
        width = box["width"] * image["pixel_width"]
        height = box["height"] * image["pixel_height"]
        candidates.append({
            "face_index": index,
            "face_box": {key: rounded(value) for key, value in box.items()},
            "face_pixel_width": rounded(width), "face_pixel_height": rounded(height),
            "measurement_quality": measurement_quality(min(width, height)),
        })
    return candidates


def build_output(images: list[dict], contract: dict, face_index: int | None = None,
                 raw: bool = False, include_low_quality: bool = False,
                 reviews: dict[str, dict] | None = None) -> dict:
    reviews = reviews or {}
    rows = []
    for image in images:
        # Area orders candidates for review only. It never chooses a group subject.
        image["faces"] = sorted(image.get("faces", []), key=lambda face: (
            -face["bounding_box"]["width"] * face["bounding_box"]["height"],
            face["bounding_box"]["x"], face["bounding_box"]["y"],
        ))
        count = len(image["faces"])
        row = {key: image[key] for key in ("path", "image_sha256", "pixel_width", "pixel_height") if key in image}
        row.update(face_count=count, measurement_signature=contract["measurement_signature"])
        rows.append(row)
        if image.get("error"):
            row["error"] = image["error"]
            continue
        candidates = candidate_boxes(image)
        row["candidate_faces"] = candidates
        if image.get("detection_limit_reached"):
            row["error"] = "Detection face limit reached; increase --max-faces or manually crop before selecting"
            continue
        if count == 0:
            row["error"] = "No face detected"
            continue
        review = reviews.get(str(Path(image["path"]).resolve()))
        selected = face_index
        if review:
            if (review.get("image_sha256") != image.get("image_sha256") or
                    review.get("measurement_signature") != contract["measurement_signature"]):
                row["error"] = "Stale review: image hash or measurement signature changed; review again"
                continue
            if selected is not None and selected != review["face_index"]:
                row["error"] = "--face-index conflicts with the per-image review selection"
                continue
            selected = review["face_index"]
        if count > 1 and selected is None:
            row["error"] = "Multiple faces detected; visually confirm the subject and rerun with --face-index N or --review-file"
            continue
        selected = 0 if selected is None else selected
        if selected < 0 or selected >= count:
            row["error"] = f"Face index {selected} not found"
            continue
        if review and review["face_box"] != candidates[selected]["face_box"]:
            row["error"] = "Stale review: selected face box changed; review again"
            continue
        face = image["faces"][selected]
        raw_metrics = face_metrics(face)
        row.update({key: value for key, value in candidates[selected].items() if key != "face_index"})
        row.update(
            selected_face_index=selected,
            metrics={key: rounded(value) for key, value in raw_metrics.items()},
            profile_metrics=profile_metrics(raw_metrics),
            quality_flags=face.get("quality_flags", []),
            subject_confirmed=bool(review and review["subject_confirmed"] is True),
            canonical_approved=bool(review and review["canonical_approved"] is True),
            human_review=review,
        )
        row["canonical_eligible"] = (
            row["subject_confirmed"] and row["canonical_approved"] and
            row["measurement_quality"] != "low" and not row["quality_flags"]
        )
        if raw:
            row["landmarks"] = {
                name: [[rounded(x), rounded(y)] for x, y in decode_points(region)]
                for name, region in face["landmarks"].items()
            }
            if "raw_mesh" in face:
                row["mediapipe_landmarks"] = face["raw_mesh"]
                row["mediapipe_landmark_space"] = "image-normalized x-right/y-down; z is model-relative, not metric depth"
    valid = [row for row in rows if "metrics" in row and not row["quality_flags"] and
             (include_low_quality or row["measurement_quality"] != "low")]
    canonical = [row for row in rows if row.get("canonical_eligible")]
    canonical_profile = summarize([{"metrics": row["profile_metrics"]} for row in canonical])
    neutral_fields = {"eye_height_over_eye_width", "mouth_width_over_face_width",
                      "mouth_width_over_eye_center_distance", "mouth_height_over_mouth_width"}
    canonical_profile = {("neutral_" + key if key in neutral_fields else key): value
                         for key, value in canonical_profile.items()}
    mp = contract["detector"]["name"] == "mediapipe"
    return {
        "schema_version": "2.0.0",
        "status": "invalid-selection" if any("error" in row for row in rows) else "ok",
        **contract,
        "runtime": {"python": platform.python_version(), "platform": platform.platform()},
        "review_scope": {key: next(iter(reviews.values()))[key] for key in ("subject_id", "era_id")} if reviews else None,
        "normalization": "All landmark coordinates and distances use the detected face box as 1.0 × 1.0.",
        "caution": "Legacy unit-square x/W,y/H geometry is anisotropic: Euclidean distances and aspect/tilt values are not physical pixel-space ratios. Same-named metrics from different backends are NOT equivalent. Geometry is not identity verification.",
        "summary_policy": "summary is descriptive, not canonical; excludes border faces and faces under 96 px unless --include-low-quality. canonical_summary always excludes low quality and requires hash-bound human confirmation plus canonical approval.",
        "profile_field_mapping": "Only canonical_profile_summary adds neutral_ prefixes after human approval. Observed min/max is not a calibrated 95% acceptance envelope.",
        "metric_compatibility": {
            "mapped_metrics": "backend-specific; compare only identical measurement_signature after project calibration",
            "eye_centers": "mean eye rims; no pupils/iris fallback" if mp else "Vision pupils or eye-rim fallback",
            "nose_width": "MediaPipe indices 98,327 alar proxy" if mp else "Vision nose-region horizontal extent",
            "unavailable_metrics": ["yaw_degrees", "roll_degrees", "pitch_degrees", "contour_width", "contour_height", "contour_aspect", "lower_jaw_width", "jaw_taper"] if mp else [],
            "unavailable_profile_metrics": ["lower_jaw_width_over_contour_width", "vision_lower_face_contour_height_over_width"] if mp else [],
            "pose_status": "null; no validated Euler conversion from MediaPipe effect matrices" if mp else "Vision native pose estimates",
            "facs_or_identity_scores": "not produced",
        },
        "images": rows,
        "summary": summarize(valid),
        "canonical_summary": summarize(canonical),
        "canonical_profile_summary": canonical_profile,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("images", nargs="+")
    parser.add_argument("--backend", choices=("auto", "vision", "mediapipe"), default="auto",
                        help="auto keeps Vision on macOS and uses MediaPipe elsewhere; never falls back silently")
    parser.add_argument("--model", type=Path, help="Local MediaPipe face_landmarker.task (required for MediaPipe)")
    parser.add_argument("--vision-cpu-only", action="store_true", help="Explicit CPU-only mode for legacy Vision in headless macOS environments")
    parser.add_argument("--max-faces", type=int, default=10, help="MediaPipe detection capacity (at least 2; default 10, not an identity gate)")
    parser.add_argument("--detection-confidence", type=float, default=0.5, help="MediaPipe documented detection default, not an identity score")
    parser.add_argument("--presence-confidence", type=float, default=0.5, help="MediaPipe documented landmark presence default")
    parser.add_argument("--face-index", type=int, help="Human-selected area-sorted index; use per-image reviews when subjects move")
    parser.add_argument("--review-file", help="Hash-bound per-image human selections and canonical approval")
    parser.add_argument("--raw", action="store_true", help="Include decoded landmarks, plus the native MediaPipe mesh")
    parser.add_argument("--include-low-quality", action="store_true", help="Include <96 px in descriptive summary only")
    parser.add_argument("--output", type=Path, help="Write UTF-8 JSON without shell encoding differences")
    args = parser.parse_args(argv)
    if args.max_faces < 2:
        parser.error("--max-faces must be >=2 so group scenes are not silently reduced to one face")
    if args.face_index is not None and args.face_index < 0:
        parser.error("--face-index must be nonnegative")
    for value in (args.detection_confidence, args.presence_confidence):
        if not math.isfinite(value) or not 0 <= value <= 1:
            parser.error("confidence options must be finite numbers in [0,1]")
    backend_name = ("vision" if sys.platform == "darwin" else "mediapipe") if args.backend == "auto" else args.backend
    try:
        reviews = load_reviews(args.review_file)
        if backend_name == "mediapipe":
            if args.vision_cpu_only:
                raise BackendError("--vision-cpu-only applies only to --backend vision")
            if args.model is None:
                raise BackendError("MediaPipe requires --model PATH. See references/measurement-backends.md for installation and model download.")
            backend = MediaPipeBackend(args.model, args.max_faces, args.detection_confidence, args.presence_confidence)
            try:
                contract = measurement_contract(backend.metadata)
                images = backend.extract(args.images)
            finally:
                backend.close()
        else:
            if args.model is not None:
                raise BackendError("--model applies only to MediaPipe; select --backend mediapipe explicitly on macOS")
            images, metadata = extract_vision(args.images, args.vision_cpu_only)
            contract = measurement_contract(metadata)
        output = build_output(images, contract, args.face_index, args.raw, args.include_low_quality, reviews)
        exit_code = 2 if output["status"] != "ok" else 0
    except (BackendError, OSError, ValueError) as error:
        output = {"schema_version": "2.0.0", "status": "backend-error", "backend": backend_name,
                  "error": str(error), "images": [], "summary": {}, "canonical_summary": {},
                  "canonical_profile_summary": {}}
        exit_code = 1
    serialized = json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
    if args.output:
        try:
            args.output.write_text(serialized + "\n", encoding="utf-8")
        except OSError as error:
            sys.stderr.write(f"Cannot write output: {error}\n")
            return 1
    else:
        print(serialized)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
