#!/usr/bin/env python3
"""Validate the structural integrity of an identity-profile JSON file."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from pathlib import Path


REQUIRED_TOP_LEVEL = {
    "schema_version",
    "profile_status",
    "subject",
    "rights",
    "era_states",
    "stable_identity_features",
    "geometry",
    "expression_prototypes",
    "prompt_anchor",
    "gallery_plan",
    "generation_recipe_policy",
    "qc",
    "known_gaps",
}

ALLOWED_PROFILE_STATUSES = {"draft", "research-baseline", "production-ready"}
SUPPORTED_SCHEMA_VERSION = "1.0.0"
ALLOWED_CONFIDENCE = {"high", "medium", "low"}
ALLOWED_RIGHTS_STATUSES = {"research-index-only", "conditional", "production-authorized", "rejected"}
GALLERY_NAMES = ("conditioning_gallery", "expression_gallery", "training_set", "holdout_set", "hard_negatives")
REQUIRED_MANIFEST_COLUMNS = {
    "source_id",
    "local_file",
    "page_url",
    "direct_asset_url",
    "source_title",
    "source_owner",
    "date",
    "era",
    "scene",
    "yaw_band",
    "expression",
    "face_pixel_min",
    "occlusion",
    "group_image",
    "subject_box",
    "verification_status",
    "rights_status",
    "likeness_status",
    "ai_training_status",
    "derivative_output_status",
    "commercial_status",
    "contract_reference",
    "sha256",
    "perceptual_hash",
    "accepted_role",
    "rejection_reason",
    "notes",
}
PRODUCTION_MANIFEST_COLUMNS = {
    "reference_conditioning_status",
    "biometric_processing_status",
    "territory",
    "term",
    "exclusivity_status",
    "endorsement_status",
    "attribution_requirements",
    "share_alike_requirements",
    "model_dependency_status",
    "rights_expiry",
    "rights_review_date",
}


def manifest_data(path: Path) -> tuple[dict[str, dict[str, str]], set[str], set[str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        by_id: dict[str, dict[str, str]] = {}
        duplicates: set[str] = set()
        for row in rows:
            source_id = str(row.get("source_id") or "").strip()
            if not source_id:
                continue
            if source_id in by_id:
                duplicates.add(source_id)
            by_id[source_id] = row
        return by_id, set(reader.fieldnames or []), duplicates


def is_sha256(value: object) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-fA-F]{64}", value) is not None


def is_affirmative(value: object) -> bool:
    return str(value or "").strip().lower() in {
        "authorized",
        "permitted",
        "approved",
        "true",
        "yes",
        "production-authorized",
    }


def is_denial(value: object) -> bool:
    normalized = str(value or "").strip().lower().replace("_", "-")
    return normalized in {
        "denied",
        "prohibited",
        "false",
        "no",
        "not-authorized",
        "not-permitted",
        "research-only",
        "non-commercial",
        "expired",
    }


def is_cleared_status(value: object) -> bool:
    return str(value or "").strip().lower() in {"cleared", "not-applicable", "n/a"}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def referenced_source_ids(value: object) -> set[str]:
    found: set[str] = set()
    if isinstance(value, dict):
        for key, item in value.items():
            if key == "source_ids" and isinstance(item, list):
                found.update(str(source_id) for source_id in item)
            else:
                found.update(referenced_source_ids(item))
    elif isinstance(value, list):
        for item in value:
            found.update(referenced_source_ids(item))
    return found


def range_errors(value: object, path: str = "$") -> list[str]:
    errors: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            errors.extend(range_errors(item, f"{path}.{key}"))
    elif isinstance(value, list):
        if len(value) == 2 and all(isinstance(item, (int, float)) for item in value):
            if value[0] > value[1]:
                errors.append(f"{path}: range minimum exceeds maximum")
        else:
            for index, item in enumerate(value):
                errors.extend(range_errors(item, f"{path}[{index}]"))
    return errors


def validate(profile: dict, manifest: Path | None) -> list[str]:
    errors = []
    manifest_rows: dict[str, dict[str, str]] = {}
    manifest_columns: set[str] = set()
    manifest_duplicates: set[str] = set()
    if manifest:
        try:
            manifest_rows, manifest_columns, manifest_duplicates = manifest_data(manifest)
        except OSError as error:
            errors.append(f"Cannot read manifest: {error}")

    missing = sorted(REQUIRED_TOP_LEVEL - profile.keys())
    if missing:
        errors.append(f"Missing top-level fields: {', '.join(missing)}")
    if profile.get("schema_version") != SUPPORTED_SCHEMA_VERSION:
        errors.append(f"schema_version must be {SUPPORTED_SCHEMA_VERSION}")

    subject = profile.get("subject", {})
    if not subject.get("subject_id"):
        errors.append("subject.subject_id is required")

    era_ids = [era.get("era_id") for era in profile.get("era_states", [])]
    if len(era_ids) != len(set(era_ids)):
        errors.append("era_states contains duplicate era_id values")
    default_era = subject.get("default_target_era")
    if default_era and default_era not in era_ids:
        errors.append("subject.default_target_era does not match any era_states.era_id")

    status = profile.get("profile_status")
    if status not in ALLOWED_PROFILE_STATUSES:
        errors.append(f"profile_status must be one of: {', '.join(sorted(ALLOWED_PROFILE_STATUSES))}")

    for name, record in profile.get("stable_identity_features", {}).items():
        if not isinstance(record, dict) or "value" not in record:
            continue
        if record.get("confidence") not in ALLOWED_CONFIDENCE:
            errors.append(f"stable_identity_features.{name}.confidence must be high, medium, or low")
        if not record.get("source_ids"):
            errors.append(f"stable_identity_features.{name}.source_ids is required")
        if not record.get("positive_anchors"):
            errors.append(f"stable_identity_features.{name}.positive_anchors is required")
        if not record.get("negative_anchors"):
            errors.append(f"stable_identity_features.{name}.negative_anchors is required")

    expression_ids = [item.get("expression_id") for item in profile.get("expression_prototypes", [])]
    if len(expression_ids) != len(set(expression_ids)):
        errors.append("expression_prototypes contains duplicate expression_id values")
    priority_types = {
        type(item.get("priority")).__name__
        for item in profile.get("expression_prototypes", [])
        if item.get("priority") is not None
    }
    if len(priority_types) > 1:
        errors.append("expression_prototypes.priority values must use one consistent type")
    for index, expression in enumerate(profile.get("expression_prototypes", [])):
        prefix = f"expression_prototypes[{index}]"
        if not expression.get("source_ids"):
            errors.append(f"{prefix}.source_ids is required")
        if not expression.get("evidence_status"):
            errors.append(f"{prefix}.evidence_status is required")
        for field in ("visual_cues", "anti_cues", "head_pose_range_degrees", "gaze", "mouth_state"):
            if field not in expression:
                errors.append(f"{prefix}.{field} is required")
        for au in expression.get("active_aus") or []:
            if not isinstance(au, str) or not re.fullmatch(r"AU\d+", au):
                errors.append(f"{prefix}.active_aus must contain separate AU identifiers such as AU6")
        facs = expression.get("facs_0_to_1")
        if isinstance(facs, dict) and facs:
            measurement_status = str(expression.get("facs_measurement_status", "")).lower()
            if not measurement_status or "not-measured" in measurement_status:
                errors.append(f"{prefix}.facs_0_to_1 requires measured provenance; otherwise set it to null")
            for au in facs:
                if not re.fullmatch(r"AU\d+", str(au)):
                    errors.append(f"{prefix}.facs_0_to_1 keys must be separate AU identifiers such as AU6")
        blendshapes = expression.get("blendshape_targets")
        if isinstance(blendshapes, dict) and blendshapes:
            measurement_status = str(expression.get("blendshape_measurement_status", "")).lower()
            if not measurement_status or "not-measured" in measurement_status:
                errors.append(f"{prefix}.blendshape_targets requires measured provenance; otherwise set it to null")
        landmark_targets = expression.get("landmark_targets")
        if isinstance(landmark_targets, dict) and landmark_targets and not expression.get("landmark_target_provenance"):
            errors.append(f"{prefix}.landmark_targets requires landmark_target_provenance; keep observed samples separate")

    required_galleries = set(GALLERY_NAMES)
    missing_galleries = sorted(required_galleries - profile.get("gallery_plan", {}).keys())
    if missing_galleries:
        errors.append(f"gallery_plan missing: {', '.join(missing_galleries)}")

    if profile.get("qc", {}).get("identity_matchers_required", 0) < 2:
        errors.append("qc.identity_matchers_required must be at least 2")

    geometry = profile.get("geometry", {})
    if geometry.get("canonical_near_frontal_ranges"):
        detector = geometry.get("detector") or {}
        for field in ("name", "version", "preprocessing_hash"):
            if not detector.get(field):
                errors.append(f"geometry.detector.{field} is required when canonical ranges are populated")

    errors.extend(range_errors(profile))

    if manifest:
        missing_columns = sorted(REQUIRED_MANIFEST_COLUMNS - manifest_columns)
        if missing_columns:
            errors.append(f"Manifest missing required columns: {', '.join(missing_columns)}")
        if manifest_duplicates:
            errors.append(f"Manifest contains duplicate source IDs: {', '.join(sorted(manifest_duplicates))}")
        unknown = sorted(referenced_source_ids(profile) - set(manifest_rows))
        if unknown:
            errors.append(f"Profile references source IDs missing from manifest: {', '.join(unknown)}")
        for source_id, row in manifest_rows.items():
            rights_status = str(row.get("rights_status") or "").strip()
            if rights_status not in ALLOWED_RIGHTS_STATUSES:
                errors.append(
                    f"Manifest {source_id}.rights_status must be one of: "
                    f"{', '.join(sorted(ALLOWED_RIGHTS_STATUSES))}"
                )
            sha256 = str(row.get("sha256") or "").strip()
            if sha256 and not is_sha256(sha256):
                errors.append(f"Manifest {source_id}.sha256 must be exactly 64 hexadecimal characters")
            verification = str(row.get("verification_status") or "").lower()
            if "landmark-sample-recorded" in verification:
                for field in ("subject_box", "face_pixel_min", "sha256", "perceptual_hash"):
                    if not str(row.get(field) or "").strip():
                        errors.append(f"Manifest {source_id}.{field} is required for a landmark sample")
                if not str(row.get("local_file") or row.get("direct_asset_url") or "").strip():
                    errors.append(
                        f"Manifest {source_id} landmark sample requires local_file or exact direct_asset_url"
                    )

    if status == "production-ready":
        rights = profile.get("rights", {})
        galleries = profile.get("gallery_plan", {})
        qc = profile.get("qc", {})
        for feature_name in ("face_shape", "cheeks", "jaw_and_chin", "eyebrows", "eyes", "nose", "mouth"):
            record = profile.get("stable_identity_features", {}).get(feature_name)
            if not isinstance(record, dict) or not record.get("value") or not record.get("source_ids"):
                errors.append(f"production-ready requires a sourced stable_identity_features.{feature_name}")
        if not profile.get("expression_prototypes"):
            errors.append("production-ready requires at least one sourced expression prototype")
        if not profile.get("geometry", {}).get("canonical_near_frontal_ranges"):
            errors.append("production-ready requires calibrated canonical geometry ranges")
        if rights.get("production_authorized") is not True:
            errors.append("production-ready requires rights.production_authorized=true")
        scope = rights.get("production_authorized_scope")
        if not isinstance(scope, dict):
            errors.append("production-ready requires rights.production_authorized_scope")
            scope = {}
        for field in (
            "likeness",
            "reference_conditioning",
            "biometric_processing",
            "derivative_outputs",
            "commercial_advertising",
        ):
            if scope.get(field) is not True:
                errors.append(f"production-ready requires rights.production_authorized_scope.{field}=true")
        for field in ("territory", "term"):
            if not str(scope.get(field) or "").strip() or is_denial(scope.get(field)):
                errors.append(f"production-ready requires a cleared rights.production_authorized_scope.{field}")
        if galleries.get("training_set", {}).get("source_ids") and scope.get("ai_training") is not True:
            errors.append("production-ready with a training_set requires rights.production_authorized_scope.ai_training")
        for gallery in ("conditioning_gallery", "expression_gallery", "holdout_set", "hard_negatives"):
            if not galleries.get(gallery, {}).get("source_ids"):
                errors.append(f"production-ready requires non-empty gallery_plan.{gallery}.source_ids")
        if galleries.get("conditioning_gallery", {}).get("target_era_coverage_status") != "complete":
            errors.append("production-ready requires conditioning_gallery.target_era_coverage_status=complete")
        if not galleries.get("conditioning_gallery", {}).get("approved_shot_list_reference"):
            errors.append("production-ready requires conditioning_gallery.approved_shot_list_reference")
        if galleries.get("leakage_audit_passed") is not True:
            errors.append("production-ready requires gallery_plan.leakage_audit_passed=true")

        gallery_ids = {
            name: {str(item) for item in galleries.get(name, {}).get("source_ids", [])}
            for name in GALLERY_NAMES
        }
        clusters = galleries.get("identity_clusters", [])
        if not clusters:
            errors.append("production-ready requires non-empty gallery_plan.identity_clusters")
        cluster_ids: list[str] = []
        for index, cluster in enumerate(clusters):
            prefix = f"gallery_plan.identity_clusters[{index}]"
            if not isinstance(cluster, dict):
                errors.append(f"{prefix} must be an object")
                continue
            cluster_id = str(cluster.get("cluster_id") or "").strip()
            cluster_ids.append(cluster_id)
            for field in ("cluster_id", "era_id", "yaw_band", "appearance_state", "medoid_source_id"):
                if not str(cluster.get(field) or "").strip():
                    errors.append(f"{prefix}.{field} is required")
            if str(cluster.get("era_id") or "") != str(default_era or ""):
                errors.append(f"{prefix}.era_id must match subject.default_target_era")
            source_ids = {str(item) for item in cluster.get("source_ids", [])}
            if not source_ids:
                errors.append(f"{prefix}.source_ids is required")
            if not source_ids <= gallery_ids["conditioning_gallery"]:
                errors.append(f"{prefix}.source_ids must belong to conditioning_gallery")
            if str(cluster.get("medoid_source_id") or "") not in source_ids:
                errors.append(f"{prefix}.medoid_source_id must belong to the cluster source_ids")
            for field in ("embedding_model_hash", "preprocessing_hash", "centroid_sha256"):
                if not is_sha256(str(cluster.get(field) or "")):
                    errors.append(f"{prefix}.{field} must be a SHA-256")
            centroid_file = str(cluster.get("centroid_file") or "").strip()
            if not centroid_file:
                errors.append(f"{prefix}.centroid_file is required")
            else:
                centroid_path = Path(centroid_file)
                if not centroid_path.is_absolute() and manifest:
                    centroid_path = manifest.parent / centroid_path
                if not centroid_path.is_file():
                    errors.append(f"{prefix}.centroid_file does not exist: {centroid_path}")
                elif is_sha256(str(cluster.get("centroid_sha256") or "")):
                    if file_sha256(centroid_path) != str(cluster.get("centroid_sha256") or "").lower():
                        errors.append(f"{prefix}.centroid_sha256 does not match centroid_file")
        if len(cluster_ids) != len(set(cluster_ids)):
            errors.append("gallery_plan.identity_clusters must have unique cluster_id values")
        for index, left_name in enumerate(GALLERY_NAMES):
            for right_name in GALLERY_NAMES[index + 1:]:
                overlap = sorted(gallery_ids[left_name] & gallery_ids[right_name])
                if overlap:
                    errors.append(
                        f"Gallery leakage between {left_name} and {right_name}: {', '.join(overlap)}"
                    )
        if not manifest:
            errors.append("production-ready requires --manifest")
        else:
            missing_production_columns = sorted(PRODUCTION_MANIFEST_COLUMNS - manifest_columns)
            if missing_production_columns:
                errors.append(
                    "Production manifest missing rights columns: "
                    + ", ".join(missing_production_columns)
                )
            subject_gallery_names = ("conditioning_gallery", "expression_gallery", "training_set", "holdout_set")
            subject_ids = set().union(*(gallery_ids[name] for name in subject_gallery_names))
            all_production_ids = subject_ids | gallery_ids["hard_negatives"]
            orphan_references = sorted(referenced_source_ids(profile) - all_production_ids)
            if orphan_references:
                errors.append(
                    "production-ready source references must belong to an explicit gallery: "
                    + ", ".join(orphan_references)
                )
            gallery_hashes = {
                name: {
                    str(manifest_rows[source_id].get("sha256") or "")
                    for source_id in gallery_ids[name]
                    if source_id in manifest_rows and str(manifest_rows[source_id].get("sha256") or "")
                }
                for name in GALLERY_NAMES
            }
            for index, left_name in enumerate(GALLERY_NAMES):
                for right_name in GALLERY_NAMES[index + 1:]:
                    overlap = sorted(gallery_hashes[left_name] & gallery_hashes[right_name])
                    if overlap:
                        errors.append(
                            f"Gallery asset-hash leakage between {left_name} and {right_name}: "
                            + ", ".join(overlap)
                        )
            for source_id in sorted(referenced_source_ids(profile)):
                row = manifest_rows.get(source_id)
                if row and str(row.get("rights_status") or "") != "production-authorized":
                    errors.append(
                        f"production-ready profile references non-production-authorized source {source_id}"
                    )
            for source_id in sorted(all_production_ids):
                row = manifest_rows.get(source_id)
                if not row:
                    continue
                prefix = f"Production manifest {source_id}"
                if str(row.get("rights_status") or "") != "production-authorized":
                    errors.append(f"{prefix}.rights_status must be production-authorized")
                if not str(row.get("contract_reference") or "").strip():
                    errors.append(f"{prefix}.contract_reference is required")
                if not is_affirmative(row.get("biometric_processing_status")):
                    errors.append(f"{prefix}.biometric_processing_status must be affirmative")
                for field in (
                    "territory",
                    "term",
                    "attribution_requirements",
                    "share_alike_requirements",
                    "rights_expiry",
                    "rights_review_date",
                ):
                    if not str(row.get(field) or "").strip() or is_denial(row.get(field)):
                        errors.append(f"{prefix}.{field} is required")
                for field in ("exclusivity_status", "endorsement_status", "model_dependency_status"):
                    if not is_cleared_status(row.get(field)):
                        errors.append(f"{prefix}.{field} must be cleared or not-applicable")
                local_file = str(row.get("local_file") or "").strip()
                if not local_file:
                    errors.append(f"{prefix}.local_file is required")
                else:
                    local_path = Path(local_file)
                    if not local_path.is_absolute() and manifest:
                        local_path = manifest.parent / local_path
                    if not local_path.is_file():
                        errors.append(f"{prefix}.local_file does not exist: {local_path}")
                    elif is_sha256(str(row.get("sha256") or "").strip()):
                        actual_hash = file_sha256(local_path)
                        if actual_hash.lower() != str(row.get("sha256") or "").strip().lower():
                            errors.append(f"{prefix}.sha256 does not match local_file")
                if not is_sha256(str(row.get("sha256") or "").strip()):
                    errors.append(f"{prefix}.sha256 must be a valid file hash")
                if not str(row.get("perceptual_hash") or "").strip():
                    errors.append(f"{prefix}.perceptual_hash is required")
                if str(row.get("group_image") or "").strip().lower() == "true" and not str(row.get("subject_box") or "").strip():
                    errors.append(f"{prefix}.subject_box is required for a group image")
                if "verified" not in str(row.get("verification_status") or "").lower():
                    errors.append(f"{prefix}.verification_status must show manual verification")
                if source_id in subject_ids:
                    if str(row.get("era") or "") != str(default_era or ""):
                        errors.append(f"{prefix}.era must match subject.default_target_era")
                    for field in ("likeness_status", "derivative_output_status", "commercial_status"):
                        if not is_affirmative(row.get(field)):
                            errors.append(f"{prefix}.{field} must be affirmative")
                if source_id in gallery_ids["conditioning_gallery"] | gallery_ids["expression_gallery"]:
                    if not is_affirmative(row.get("reference_conditioning_status")):
                        errors.append(f"{prefix}.reference_conditioning_status must be affirmative")
                if source_id in gallery_ids["training_set"] and not is_affirmative(row.get("ai_training_status")):
                    errors.append(f"{prefix}.ai_training_status must be affirmative")

        matchers = qc.get("identity_matchers", [])
        matcher_ids: list[str] = []
        matcher_families: list[str] = []
        matcher_hashes: list[str] = []
        if len(matchers) < 2:
            errors.append("production-ready requires at least two configured identity matchers")
        for index, matcher in enumerate(matchers):
            prefix = f"qc.identity_matchers[{index}]"
            if not isinstance(matcher, dict):
                errors.append(f"{prefix} must be an object with independent model provenance")
                continue
            matcher_id = str(matcher.get("matcher_id") or "").strip()
            family = str(matcher.get("model_family") or "").strip()
            model_hash = str(matcher.get("model_hash") or "").strip()
            matcher_ids.append(matcher_id)
            matcher_families.append(family)
            matcher_hashes.append(model_hash)
            if not matcher_id:
                errors.append(f"{prefix}.matcher_id is required")
            if not family:
                errors.append(f"{prefix}.model_family is required")
            if not is_sha256(model_hash):
                errors.append(f"{prefix}.model_hash must be a SHA-256")
            if not is_sha256(str(matcher.get("preprocessing_hash") or "")):
                errors.append(f"{prefix}.preprocessing_hash must be a SHA-256")
        if len(matcher_ids) != len(set(matcher_ids)):
            errors.append("production-ready matchers must have distinct matcher_id values")
        if len(matcher_families) != len(set(matcher_families)):
            errors.append("production-ready matchers must use distinct model_family values")
        if len(matcher_hashes) != len(set(matcher_hashes)):
            errors.append("production-ready matchers must use distinct model hashes")

        thresholds = qc.get("calibrated_thresholds", [])
        threshold_matcher_id_list: list[str] = []
        if len(thresholds) < 2:
            errors.append("production-ready requires calibrated thresholds for both matchers")
        for index, threshold in enumerate(thresholds):
            prefix = f"qc.calibrated_thresholds[{index}]"
            if not isinstance(threshold, dict):
                errors.append(f"{prefix} must include matcher_id, threshold, risk point, and calibration hash")
                continue
            matcher_id = str(threshold.get("matcher_id") or "").strip()
            threshold_matcher_id_list.append(matcher_id)
            if not isinstance(threshold.get("threshold"), (int, float)):
                errors.append(f"{prefix}.threshold must be numeric")
            false_match_rate = threshold.get("target_false_match_rate")
            if not isinstance(false_match_rate, (int, float)) or not 0 < false_match_rate < 1:
                errors.append(f"{prefix}.target_false_match_rate must be between 0 and 1")
            if not is_sha256(str(threshold.get("calibration_hash") or "")):
                errors.append(f"{prefix}.calibration_hash must be a SHA-256")
        if len(threshold_matcher_id_list) != len(set(threshold_matcher_id_list)):
            errors.append("production-ready calibrated thresholds must have unique matcher_id values")
        if set(matcher_ids) != set(threshold_matcher_id_list):
            errors.append("production-ready requires exactly one calibrated threshold per matcher_id")
        if qc.get("genuine_pair_count", 0) <= 0 or qc.get("impostor_pair_count", 0) <= 0:
            errors.append("production-ready requires positive genuine and impostor pair counts")
        if qc.get("holdout_isolated") is not True:
            errors.append("production-ready requires qc.holdout_isolated=true")
        for field in ("calibration_dataset_hash", "expression_calibration_hash", "geometry_envelope_hash"):
            if not is_sha256(str(qc.get(field) or "")):
                errors.append(f"production-ready requires qc.{field} as SHA-256")
        if qc.get("expression_gate_status") != "calibrated":
            errors.append("production-ready requires qc.expression_gate_status=calibrated")
        if qc.get("geometry_envelope_status") != "calibrated":
            errors.append("production-ready requires qc.geometry_envelope_status=calibrated")

    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("profile", type=Path)
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()

    try:
        profile = json.loads(args.profile.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        print(f"INVALID: {error}", file=sys.stderr)
        return 1

    errors = validate(profile, args.manifest)
    if errors:
        for error in errors:
            print(f"INVALID: {error}", file=sys.stderr)
        return 1
    print("VALID")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
