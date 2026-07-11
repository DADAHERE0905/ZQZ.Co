#!/usr/bin/env python3
"""Measure normalized face-landmark ratios with macOS Vision.

The script does not identify people. It measures landmarks in user-selected images.
All distances are normalized to the detected face bounding box.
"""

from __future__ import annotations

import argparse
import base64
import json
import math
import statistics
import struct
import subprocess
import sys
from pathlib import Path


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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("images", nargs="+")
    parser.add_argument(
        "--face-index",
        type=int,
        default=None,
        help="Manually confirmed face index after sorting by area; required when an image contains multiple faces",
    )
    parser.add_argument("--raw", action="store_true", help="Include decoded landmark points")
    parser.add_argument("--include-low-quality", action="store_true", help="Include faces under 96 px in the aggregate summary")
    args = parser.parse_args()

    script = Path(__file__).with_name("extract_face_landmarks.jxa")
    command = ["osascript", "-l", "JavaScript", str(script), *args.images]
    process = subprocess.run(command, check=False, capture_output=True, text=True)
    if process.returncode:
        sys.stderr.write(process.stderr)
        return process.returncode
    try:
        images = json.loads(process.stdout)
    except json.JSONDecodeError as error:
        sys.stderr.write(f"Cannot parse Vision output: {error}\n{process.stdout}\n{process.stderr}")
        return 1

    rows = []
    for image in images:
        face_count = len(image.get("faces", []))
        if image.get("error"):
            rows.append({
                "path": image["path"],
                "error": image["error"],
            })
            continue
        if face_count == 0:
            rows.append({"path": image["path"], "error": "No face detected"})
            continue
        if face_count > 1 and args.face_index is None:
            rows.append({
                "path": image["path"],
                "face_count": face_count,
                "error": "Multiple faces detected; visually confirm the subject and rerun with --face-index N",
            })
            continue
        selected_face_index = 0 if args.face_index is None else args.face_index
        if selected_face_index < 0 or selected_face_index >= face_count:
            rows.append({
                "path": image["path"],
                "face_count": face_count,
                "error": f"Face index {selected_face_index} not found",
            })
            continue
        face = image["faces"][selected_face_index]
        face_pixel_width = face["bounding_box"]["width"] * image["pixel_width"]
        face_pixel_height = face["bounding_box"]["height"] * image["pixel_height"]
        face_pixel_min = min(face_pixel_width, face_pixel_height)
        raw_metrics = face_metrics(face)
        row = {
            "path": image["path"],
            "pixel_width": image["pixel_width"],
            "pixel_height": image["pixel_height"],
            "face_count": face_count,
            "selected_face_index": selected_face_index,
            "face_box": {key: rounded(value) for key, value in face["bounding_box"].items()},
            "face_pixel_width": rounded(face_pixel_width),
            "face_pixel_height": rounded(face_pixel_height),
            "measurement_quality": measurement_quality(face_pixel_min),
            "metrics": {key: rounded(value) for key, value in raw_metrics.items()},
            "profile_metrics": profile_metrics(raw_metrics),
        }
        if args.raw:
            row["landmarks"] = {
                name: [[rounded(x), rounded(y)] for x, y in decode_points(region)]
                for name, region in face["landmarks"].items()
            }
        rows.append(row)

    valid_rows = [
        row for row in rows
        if "metrics" in row and (args.include_low_quality or row["measurement_quality"] != "low")
    ]
    output = {
        "status": "invalid-selection" if any("error" in row for row in rows) else "ok",
        "normalization": "All landmark coordinates and distances use the detected face box as 1.0 × 1.0.",
        "caution": "Use only manually verified, comparable-angle images. Pose and expression can shift these 2D ratios.",
        "summary_policy": "Aggregate summary excludes detected faces under 96 px unless --include-low-quality is set.",
        "profile_field_mapping": "profile_metrics uses identity-profile schema names; add the neutral_ prefix only after the frame passes the neutral/low-expression canonical gate.",
        "images": rows,
        "summary": summarize(valid_rows),
    }
    print(json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True))
    return 2 if any("error" in row for row in rows) else 0


if __name__ == "__main__":
    raise SystemExit(main())
