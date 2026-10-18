#!/usr/bin/env python3
"""Compare reviewed canonical geometry across shots, never infer identity.

Without an owner-approved calibrated envelope, report deltas only. A geometry
pass is not an identity, expression, temporal continuity or production pass.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

NEUTRAL_FIELDS = {"eye_height_over_eye_width", "mouth_width_over_face_width",
                  "mouth_width_over_eye_center_distance", "mouth_height_over_mouth_width"}


def canonical_name(key: str) -> str:
    return "neutral_" + key if key in NEUTRAL_FIELDS else key


def compare(reference: dict, candidate: dict, envelope: dict | None = None) -> dict:
    if not isinstance(reference, dict) or not isinstance(candidate, dict) or (envelope is not None and not isinstance(envelope, dict)):
        raise ValueError("Measurement and envelope inputs must be JSON objects")
    if reference.get("status") != "ok" or not candidate.get("images"):
        raise ValueError("Reference must be a successful measurement and candidate must contain image rows")
    signature = reference.get("measurement_signature")
    if not signature or signature != candidate.get("measurement_signature"):
        raise ValueError("Incompatible measurement_signature; remeasure and recalibrate with the same backend/model/preprocessing")
    scope = reference.get("review_scope")
    if not scope or scope != candidate.get("review_scope"):
        raise ValueError("Reviewed subject_id and era_id must match across reference and candidate")
    baseline = reference.get("canonical_profile_summary")
    if not isinstance(baseline, dict) or not baseline:
        raise ValueError("Reference has no human-approved, quality-gated canonical geometry")
    if any(not isinstance(stats, dict) or type(stats.get("median")) not in (int, float) or
           not math.isfinite(stats["median"]) for stats in baseline.values()):
        raise ValueError("Reference canonical medians must be finite numbers")
    ranges = None
    if envelope is not None:
        if (envelope.get("measurement_signature") != signature or envelope.get("review_scope") != scope or
                envelope.get("calibration_status") != "calibrated" or envelope.get("owner_approved") is not True or
                not re.fullmatch(r"[0-9a-fA-F]{64}", str(envelope.get("calibration_dataset_hash", "")))):
            raise ValueError("Envelope needs matching signature/scope, calibrated status, owner_approved=true and calibration_dataset_hash")
        ranges = envelope.get("ranges")
        if not isinstance(ranges, dict) or not ranges:
            raise ValueError("Envelope requires nonempty calibrated profile-field ranges")
        for key, bounds in ranges.items():
            if (key not in baseline or not isinstance(bounds, list) or len(bounds) != 2 or
                    any(type(v) not in (float, int) or not math.isfinite(v) for v in bounds) or bounds[0] > bounds[1]):
                raise ValueError(f"Invalid or unsupported envelope range: {key}")
    rows = []
    for image in candidate.get("images", []):
        row = {"path": image["path"], "identity_verdict": "not-evaluated"}
        rows.append(row)
        if not image.get("canonical_eligible") or "error" in image:
            row.update(geometry_status="not-evaluated", reason=image.get("error") or "Requires human-confirmed neutral/near-frontal canonical approval and quality gates")
            continue
        values = {canonical_name(key): value for key, value in image["profile_metrics"].items()}
        if any(value is not None and (type(value) not in (int, float) or not math.isfinite(value)) for value in values.values()):
            raise ValueError("Candidate profile metrics must be finite numbers or null")
        row["deltas_from_reference_median"] = {
            key: round(value - baseline[key]["median"], 4)
            for key, value in values.items() if value is not None and key in baseline
        }
        if ranges is None:
            row["geometry_status"] = "diagnostic-only"
        else:
            missing = [key for key in ranges if values.get(key) is None]
            outside = [key for key, (lo, hi) in ranges.items()
                       if values.get(key) is not None and not lo <= values[key] <= hi]
            row.update(
                geometry_status="not-evaluated" if missing else "out-of-envelope" if outside else "in-envelope",
                missing_metrics=missing, outside_metrics=outside,
            )
    return {
        "status": "ok", "measurement_signature": signature, "review_scope": scope,
        "identity_verdict": "not-evaluated", "human_review_required": True,
        "caution": "Diagnostic deltas or a geometry envelope pass do not verify the same person. Retain two calibrated independent matchers and human likeness/expression review. No video tracking or identity-swap detection is performed.",
        "images": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--envelope", type=Path, help="Project-calibrated geometry envelope, never raw observed min/max")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        result = compare(json.loads(args.reference.read_text(encoding="utf-8")),
                         json.loads(args.candidate.read_text(encoding="utf-8")),
                         json.loads(args.envelope.read_text(encoding="utf-8")) if args.envelope else None)
        code = 2 if any(row["geometry_status"] in {"not-evaluated", "out-of-envelope"} for row in result["images"]) else 0
    except (OSError, ValueError, KeyError, TypeError) as error:
        result, code = {"status": "invalid-input", "error": str(error), "identity_verdict": "not-evaluated"}, 1
    content = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if args.output:
        try:
            args.output.write_text(content, encoding="utf-8")
        except OSError as error:
            parser.exit(1, f"Cannot write output: {error}\n")
    else:
        print(content, end="")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
