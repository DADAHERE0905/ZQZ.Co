"""Opt-in native inference tests. Set CFC_MODEL and CFC_IMAGE to local assets.

The public sample checks execution, not a person's identity or commercial QC.
Canonical approvals below are test fixtures, not actual human approvals.
"""

import copy
import hashlib
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
import measure_faces as measure
from mediapipe_backend import MediaPipeBackend

MODEL, IMAGE = os.environ.get("CFC_MODEL"), os.environ.get("CFC_IMAGE")


@unittest.skipUnless(MODEL and IMAGE, "Set CFC_MODEL and CFC_IMAGE for real local inference")
class InferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from PIL import Image, ImageOps
        cls.Image, cls.ImageOps = Image, ImageOps
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        cls.backend = MediaPipeBackend(Path(MODEL))
        cls.contract = measure.measurement_contract(cls.backend.metadata)
        cls.original = cls.backend.extract([str(IMAGE)])[0]
        if len(cls.original["faces"]) != 1:
            raise AssertionError(f"Smoke portrait must detect one face: {cls.original.get('error')}")
        with Image.open(IMAGE) as source:
            cls.portrait = ImageOps.exif_transpose(source).convert("RGB")

    @classmethod
    def tearDownClass(cls):
        cls.backend.close()
        cls.temp.cleanup()

    def test_real_portrait_geometry_and_hashes(self):
        result = measure.build_output([copy.deepcopy(self.original)], self.contract, raw=True)
        self.assertEqual(result["status"], "ok")
        row = result["images"][0]
        self.assertEqual(row["image_sha256"], hashlib.sha256(Path(IMAGE).read_bytes()).hexdigest())
        self.assertEqual(len(row["mediapipe_landmarks"]), 478)
        self.assertGreater(row["profile_metrics"]["eye_center_distance_over_face_width"], 0)
        self.assertIsNone(row["profile_metrics"]["vision_lower_face_contour_height_over_width"])
        self.assertFalse(result["canonical_summary"])

    def test_blank_corrupt_missing_and_unicode_paths(self):
        blank = self.root / "空白_frame.png"
        corrupt = self.root / "corrupt.jpg"
        self.Image.new("RGB", (256, 256), "white").save(blank)
        corrupt.write_bytes(b"not an image")
        data = self.backend.extract([str(blank), str(corrupt), str(self.root / "missing.png"), str(IMAGE)])
        output = measure.build_output(data, self.contract)
        self.assertEqual(output["status"], "invalid-selection")
        self.assertEqual(output["images"][0]["error"], "No face detected")
        self.assertIn("Cannot measure image", output["images"][1]["error"])
        self.assertIn("Cannot measure image", output["images"][2]["error"])
        self.assertIn("metrics", output["images"][3])

    def test_real_low_resolution_stays_out_of_summary(self):
        small = self.root / "small.png"
        self.portrait.resize((205, 256)).save(small)
        result = measure.build_output(self.backend.extract([str(small)]), self.contract)
        self.assertEqual(result["images"][0]["measurement_quality"], "low")
        self.assertFalse(result["summary"])

    def test_exif_rotation_rgb_and_unicode(self):
        path = self.root / "旋转_portrait.jpg"
        stored = self.portrait.transpose(self.Image.Transpose.ROTATE_90)
        exif = stored.getexif()
        exif[274] = 6  # Restore the 90-degree counterclockwise stored pixels.
        stored.save(path, exif=exif, quality=95)
        row = self.backend.extract([str(path)])[0]
        self.assertEqual((row["pixel_width"], row["pixel_height"]), self.portrait.size)
        self.assertEqual(len(row["faces"]), 1)

    def test_real_group_selection_and_detection_capacity(self):
        box = self.original["faces"][0]["bounding_box"]
        w, h = self.portrait.size
        x, y = box["x"] * w, (1 - box["y"] - box["height"]) * h
        fw, fh = box["width"] * w, box["height"] * h
        tile = self.portrait.crop((int(x - fw * 0.45), int(y - fh * 0.35),
                                   int(x + fw * 1.45), int(y + fh * 1.35)))
        group = self.Image.new("RGB", (tile.width * 2, tile.height), "white")
        group.paste(tile, (0, 0))
        group.paste(tile, (tile.width, 0))
        path = self.root / "group.png"
        group.save(path)
        detected = self.backend.extract([str(path)])
        self.assertEqual(len(detected[0]["faces"]), 2)
        result = measure.build_output(copy.deepcopy(detected), self.contract)
        self.assertEqual(result["status"], "invalid-selection")
        selected = measure.build_output(copy.deepcopy(detected), self.contract, face_index=1)
        self.assertEqual(selected["status"], "ok")
        self.assertEqual(selected["images"][0]["selected_face_index"], 1)
        limited = MediaPipeBackend(Path(MODEL), max_faces=2)
        try:
            output = measure.build_output(limited.extract([str(path)]), self.contract, face_index=0)
            self.assertIn("limit", output["images"][0]["error"])
        finally:
            limited.close()

    def test_cli_and_profile_to_cross_shot_workflow(self):
        # The approval is deliberately fabricated by the test, never production evidence.
        initial = self.root / "measurement.json"
        reviewed = self.root / "reviewed.json"
        reviews = self.root / "review.json"
        command = [sys.executable, str(SCRIPTS / "measure_faces.py"), "--backend", "mediapipe",
                   "--model", str(MODEL), str(IMAGE)]
        first = subprocess.run([*command, "--output", str(initial)], capture_output=True, text=True)
        self.assertEqual(first.returncode, 0, first.stderr)
        data = json.loads(initial.read_text(encoding="utf-8"))
        row = data["images"][0]
        reviews.write_text(json.dumps({
            "schema_version": "1.0", "subject_id": "smoke-fixture-only", "era_id": "sample",
            "images": [{"path": str(IMAGE), "source_id": "S-SMOKE", "image_sha256": row["image_sha256"],
                        "measurement_signature": data["measurement_signature"], "face_index": 0,
                        "face_box": row["face_box"], "subject_confirmed": True, "canonical_approved": True,
                        "reviewer": "automated-software-test-only"}],
        }), encoding="utf-8")
        second = subprocess.run([*command, "--review-file", str(reviews), "--output", str(reviewed)], capture_output=True, text=True)
        self.assertEqual(second.returncode, 0, second.stderr)
        result = json.loads(reviewed.read_text(encoding="utf-8"))
        self.assertTrue(result["canonical_profile_summary"])
        compared = subprocess.run([sys.executable, str(SCRIPTS / "compare_geometry.py"), str(reviewed), str(reviewed)], capture_output=True, text=True)
        self.assertEqual(compared.returncode, 0, compared.stderr)
        diagnostics = json.loads(compared.stdout)
        self.assertEqual(diagnostics["identity_verdict"], "not-evaluated")
        self.assertEqual(diagnostics["images"][0]["geometry_status"], "diagnostic-only")

    def test_corrupt_model_is_structured_error(self):
        invalid = self.root / "bad.task"
        invalid.write_bytes(b"invalid model")
        run = subprocess.run([sys.executable, str(SCRIPTS / "measure_faces.py"), "--backend", "mediapipe",
                              "--model", str(invalid), str(IMAGE)], capture_output=True, text=True)
        self.assertEqual(run.returncode, 1)
        self.assertEqual(json.loads(run.stdout)["status"], "backend-error")

    def test_unicode_model_path(self):
        model = self.root / "模型_face.task"
        model.write_bytes(Path(MODEL).read_bytes())
        local = MediaPipeBackend(model)
        try:
            self.assertEqual(len(local.extract([str(IMAGE)])[0]["faces"]), 1)
            self.assertEqual(local.metadata["model_sha256"], self.backend.metadata["model_sha256"])
        finally:
            local.close()

    @unittest.skipUnless(sys.platform == "darwin", "Apple Vision needs native macOS")
    def test_native_vision_entry_point(self):
        run = subprocess.run([sys.executable, str(SCRIPTS / "measure_faces.py"), "--backend", "vision", str(IMAGE)], capture_output=True, text=True)
        self.assertEqual(run.returncode, 0, run.stderr + run.stdout)
        output = json.loads(run.stdout)
        self.assertEqual(output["detector"]["name"], "vision")
        self.assertEqual(output["images"][0]["face_count"], 1)


if __name__ == "__main__":
    unittest.main()
