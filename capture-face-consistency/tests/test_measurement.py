"""Deterministic contract tests; no ML package, network or real person required."""

import copy
import io
import json
import math
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
import compare_geometry as comparison
import measure_faces as measure
import mediapipe_backend as backend
import validate_profile


def mesh():
    points = [SimpleNamespace(x=0.5, y=0.5, z=0.0) for _ in range(478)]
    for index, number in enumerate(backend.OVAL):
        theta = index * 2 * math.pi / len(backend.OVAL)
        points[number] = SimpleNamespace(x=0.5 + 0.3 * math.cos(theta),
                                         y=0.5 + 0.4 * math.sin(theta), z=0.0)
    for name, indices in backend.REGIONS.items():
        cx = 0.65 if name.startswith("left") else 0.35 if name.startswith("right") else 0.5
        cy = 0.4 if "eye" in name else 0.65
        for index, number in enumerate(indices):
            theta = index * 2 * math.pi / len(indices)
            points[number] = SimpleNamespace(x=cx + 0.05 * math.cos(theta),
                                             y=cy + 0.02 * math.sin(theta), z=0.0)
    return points


CONTRACT = {"measurement_signature": "s" * 64, "detector": {"name": "mediapipe"}}


def image(path="frame.png", size=1000, count=1):
    return {"path": path, "image_sha256": "a" * 64,
            "pixel_width": size, "pixel_height": size,
            "faces": [backend.adapt_landmarks(mesh()) for _ in range(count)]}


def review_for(source, index=0, approved=True):
    box = measure.candidate_boxes(source)[index]["face_box"]
    return {str(Path(source["path"]).resolve()): {
        "path": source["path"], "image_sha256": source["image_sha256"],
        "measurement_signature": CONTRACT["measurement_signature"],
        "face_index": index, "face_box": box, "subject_confirmed": True,
        "canonical_approved": approved, "reviewer": "unit-test-fixture",
        "subject_id": "fictional-fixture", "era_id": "design-v1", "source_id": "S001",
    }}


class MeasurementTests(unittest.TestCase):
    def test_quality_boundaries(self):
        for pixels, expected in [(0, "low"), (95.99, "low"), (96, "medium"), (159.99, "medium"), (160, "high")]:
            with self.subTest(pixels=pixels):
                self.assertEqual(measure.measurement_quality(pixels), expected)

    def test_bottom_left_coordinates_and_normalization(self):
        face = backend.adapt_landmarks(mesh())
        self.assertAlmostEqual(face["bounding_box"]["x"], 0.2)
        self.assertAlmostEqual(face["bounding_box"]["y"], 0.1)
        self.assertAlmostEqual(face["bounding_box"]["height"], 0.8)
        point = measure.decode_points(face["landmarks"]["left_eye"])[0]
        self.assertAlmostEqual(point[0], (0.7 - 0.2) / 0.6)
        self.assertAlmostEqual(point[1], (0.9 - 0.4) / 0.8)

    def test_unmappable_contour_and_pose_are_null(self):
        metrics = measure.face_metrics(backend.adapt_landmarks(mesh()))
        for key in ("yaw_degrees", "roll_degrees", "pitch_degrees", "contour_width", "contour_height", "jaw_taper"):
            self.assertIsNone(metrics[key])
        fields = measure.profile_metrics(metrics)
        self.assertIsNone(fields["vision_lower_face_contour_height_over_width"])
        self.assertIsNone(fields["lower_jaw_width_over_contour_width"])
        self.assertIsNotNone(fields["eye_center_distance_over_face_width"])

    def test_bad_topology_nonfinite_and_degenerate_mesh(self):
        cases = [mesh()[:468], mesh(), [SimpleNamespace(x=0.5, y=0.5, z=0) for _ in range(478)]]
        cases[1][0].z = float("nan")
        for points in cases:
            with self.assertRaises(ValueError):
                backend.adapt_landmarks(points)

    def test_safe_ratio_and_empty_summary(self):
        self.assertIsNone(measure.safe_ratio(1, 0))
        self.assertEqual(measure.summarize([]), {})
        self.assertEqual(measure.summarize([{"metrics": {"a": None}}]), {})

    def test_legacy_json_keys_and_raw_spaces(self):
        output = measure.build_output([image()], CONTRACT, raw=True)
        self.assertTrue({"status", "normalization", "caution", "summary_policy", "profile_field_mapping", "images", "summary"} <= output.keys())
        row = output["images"][0]
        self.assertEqual(len(row["mediapipe_landmarks"]), 478)
        self.assertIn("y-down", row["mediapipe_landmark_space"])
        self.assertEqual(output["canonical_profile_summary"], {})
        self.assertFalse(row["subject_confirmed"])

    def test_low_quality_override_cannot_enter_canonical_summary(self):
        source = image(size=100)
        reviews = review_for(source)
        output = measure.build_output([source], CONTRACT, reviews=reviews)
        self.assertEqual(output["summary"], {})
        output = measure.build_output([source], CONTRACT, include_low_quality=True, reviews=reviews)
        self.assertTrue(output["summary"])
        self.assertFalse(output["canonical_summary"])

    def test_border_faces_excluded_even_with_override(self):
        source = image()
        source["faces"][0]["quality_flags"] = ["face-touches-image-border"]
        output = measure.build_output([source], CONTRACT, include_low_quality=True, reviews=review_for(source))
        self.assertFalse(output["summary"])
        self.assertFalse(output["canonical_summary"])

    def test_multiple_faces_need_selection_and_expose_boxes(self):
        output = measure.build_output([image(count=2)], CONTRACT)
        self.assertEqual(output["status"], "invalid-selection")
        self.assertEqual(len(output["images"][0]["candidate_faces"]), 2)
        self.assertNotIn("metrics", output["images"][0])
        output = measure.build_output([image(count=2)], CONTRACT, face_index=1)
        self.assertEqual(output["images"][0]["selected_face_index"], 1)
        self.assertFalse(output["images"][0]["subject_confirmed"])

    def test_invalid_index_no_face_partial_batch_and_capacity(self):
        for index in (-1, 1, 100):
            self.assertEqual(measure.build_output([image()], CONTRACT, face_index=index)["status"], "invalid-selection")
        bad = image("bad.png", count=0)
        output = measure.build_output([bad, image()], CONTRACT)
        self.assertEqual(output["status"], "invalid-selection")
        self.assertTrue(output["summary"])
        limit = image(count=2)
        limit["detection_limit_reached"] = True
        output = measure.build_output([limit], CONTRACT, face_index=0)
        self.assertIn("limit", output["images"][0]["error"])
        self.assertFalse(output["summary"])

    def test_hash_bound_review_and_neutral_prefix(self):
        source = image()
        output = measure.build_output([source], CONTRACT, reviews=review_for(source))
        self.assertTrue(output["images"][0]["canonical_eligible"])
        self.assertIn("neutral_mouth_height_over_mouth_width", output["canonical_profile_summary"])
        for field in ("image_sha256", "measurement_signature", "face_box"):
            reviews = review_for(source)
            next(iter(reviews.values()))[field] = "stale"
            output = measure.build_output([source], CONTRACT, reviews=reviews)
            self.assertEqual(output["status"], "invalid-selection")
            self.assertFalse(output["canonical_summary"])

    def test_review_can_select_different_faces_per_image(self):
        sources = [image("one.png", count=2), image("two.png", count=2)]
        reviews = {**review_for(sources[0], 1), **review_for(sources[1], 0)}
        output = measure.build_output(sources, CONTRACT, reviews=reviews)
        self.assertEqual([r["selected_face_index"] for r in output["images"]], [1, 0])
        output = measure.build_output(sources, CONTRACT, face_index=1, reviews=reviews)
        self.assertEqual(output["status"], "invalid-selection")

    def test_review_boolean_type_and_duplicate_path_validation(self):
        entry = next(iter(review_for(image()).values()))
        payload = {"schema_version": "1.0", "subject_id": "test", "era_id": "test", "images": [entry]}
        with tempfile.TemporaryDirectory() as directory:
            file = Path(directory) / "reviews.json"
            file.write_text(json.dumps(payload), encoding="utf-8")
            self.assertEqual(len(measure.load_reviews(str(file))), 1)
            for field, value in [("subject_confirmed", "true"), ("canonical_approved", 1), ("face_index", True), ("reviewer", ""), ("source_id", "")]:
                changed = copy.deepcopy(payload)
                changed["images"][0][field] = value
                file.write_text(json.dumps(changed), encoding="utf-8")
                with self.assertRaises(ValueError):
                    measure.load_reviews(str(file))
            payload["images"].append(entry)
            file.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaises(ValueError):
                measure.load_reviews(str(file))

    def test_signature_changes_with_model_and_preprocessing(self):
        metadata = {"name": "mediapipe", "version": "0.10.21", "model_sha256": "a" * 64, "orientation": "EXIF"}
        first = measure.measurement_contract(metadata)
        for key in ("model_sha256", "orientation", "version"):
            self.assertNotEqual(first["measurement_signature"], measure.measurement_contract({**metadata, key: "changed"})["measurement_signature"])

    def test_source_hash_is_portable_across_checkout_line_endings(self):
        with tempfile.TemporaryDirectory() as directory:
            lf = Path(directory) / "unix.py"
            crlf = Path(directory) / "windows.py"
            lf.write_bytes(b"# source\nvalue = 1\n")
            crlf.write_bytes(b"# source\r\nvalue = 1\r\n")
            self.assertEqual(measure.source_sha256(lf), measure.source_sha256(crlf))

    def test_vision_requires_native_platform(self):
        with patch.object(measure.sys, "platform", "linux"), patch.object(measure.subprocess, "run") as run:
            with self.assertRaises(backend.BackendError):
                measure.extract_vision(["a.png"])
            run.assert_not_called()

    def test_cli_missing_model_and_backend_failure_are_json(self):
        stream = io.StringIO()
        with redirect_stdout(stream):
            code = measure.main(["--backend", "mediapipe", "missing.png"])
        self.assertEqual(code, 1)
        self.assertEqual(json.loads(stream.getvalue())["status"], "backend-error")
        stream = io.StringIO()
        with patch.object(measure, "MediaPipeBackend", side_effect=backend.BackendError("dependencies unavailable")), redirect_stdout(stream):
            code = measure.main(["--backend", "mediapipe", "--model", "model.task", "a.png"])
        self.assertEqual(code, 1)
        self.assertIn("dependencies", json.loads(stream.getvalue())["error"])

    def test_draft_profile_mapping_and_production_gate_remain_separate(self):
        source = image()
        output = measure.build_output([source], CONTRACT, reviews=review_for(source))
        profile = json.loads((SCRIPTS.parent / "assets/identity-profile.template.json").read_text())
        profile["subject"]["subject_id"] = "fictional-fixture"
        profile["stable_identity_features"] = {}
        profile["geometry"]["canonical_near_frontal_ranges"] = {
            key: [value["min"], value["max"]] for key, value in output["canonical_profile_summary"].items()
        }
        profile["geometry"]["detector"] = {"name": "mediapipe", "version": "test", "preprocessing_hash": "a" * 64}
        self.assertEqual(validate_profile.validate(profile, None), [])
        profile["profile_status"] = "production-ready"
        self.assertTrue(validate_profile.validate(profile, None))


class ComparisonTests(unittest.TestCase):
    def setUp(self):
        source = image()
        self.reference = measure.build_output([source], CONTRACT, reviews=review_for(source))

    def test_diagnostic_is_not_an_identity_verdict(self):
        result = comparison.compare(self.reference, self.reference)
        self.assertEqual(result["identity_verdict"], "not-evaluated")
        self.assertEqual(result["images"][0]["geometry_status"], "diagnostic-only")
        self.assertTrue(all(value == 0 for value in result["images"][0]["deltas_from_reference_median"].values()))

    def test_model_scope_and_unreviewed_reference_rejected(self):
        for field, value in [("measurement_signature", "other"), ("review_scope", {"subject_id": "other"}), ("canonical_profile_summary", {})]:
            changed = {**self.reference, field: value}
            with self.assertRaises(ValueError):
                comparison.compare(changed, self.reference)
        with self.assertRaises(ValueError):
            comparison.compare(self.reference, {**self.reference, "images": []})

    def test_approved_calibrated_envelope_only(self):
        key = "eye_center_distance_over_face_width"
        value = self.reference["canonical_profile_summary"][key]["median"]
        # Test fixture boundaries, not production thresholds.
        envelope = {"measurement_signature": CONTRACT["measurement_signature"],
                    "review_scope": self.reference["review_scope"], "calibration_status": "calibrated",
                    "owner_approved": True, "calibration_dataset_hash": "b" * 64,
                    "ranges": {key: [value - 0.01, value + 0.01]}}
        self.assertEqual(comparison.compare(self.reference, self.reference, envelope)["images"][0]["geometry_status"], "in-envelope")
        shifted = copy.deepcopy(self.reference)
        shifted["images"][0]["profile_metrics"][key] += 0.1
        self.assertEqual(comparison.compare(self.reference, shifted, envelope)["images"][0]["geometry_status"], "out-of-envelope")
        for bounds in ([2, 1], [float("nan"), 1], [True, 1], [0]):
            envelope["ranges"][key] = bounds
            with self.assertRaises(ValueError):
                comparison.compare(self.reference, self.reference, envelope)

    def test_unreviewed_candidate_is_not_evaluated(self):
        candidate = copy.deepcopy(self.reference)
        candidate["images"][0]["canonical_eligible"] = False
        self.assertEqual(comparison.compare(self.reference, candidate)["images"][0]["geometry_status"], "not-evaluated")

    def test_malformed_json_shapes_and_nonfinite_values_are_rejected(self):
        for reference, candidate in [([], self.reference), (self.reference, [])]:
            with self.assertRaises(ValueError):
                comparison.compare(reference, candidate)
        candidate = copy.deepcopy(self.reference)
        candidate["images"][0]["profile_metrics"]["eye_center_distance_over_face_width"] = float("nan")
        with self.assertRaises(ValueError):
            comparison.compare(self.reference, candidate)


if __name__ == "__main__":
    unittest.main()
