"""Local MediaPipe Tasks adapter. Imports ML/image dependencies only on demand."""

from __future__ import annotations

import base64
import hashlib
import io
import math
import struct
from pathlib import Path


class BackendError(RuntimeError):
    """An actionable backend setup or execution error."""


# Subject-relative left/right, from MediaPipe's official face mesh connections.
# Eye centers deliberately use the rim mean, never a gaze-dependent iris fallback.
REGIONS = {
    "left_eye": (263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466),
    "right_eye": (33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246),
    "left_eyebrow": (276, 283, 282, 295, 285, 300, 293, 334, 296, 336),
    "right_eyebrow": (46, 53, 52, 65, 55, 70, 63, 105, 66, 107),
    "nose": (98, 327),  # Alar-width proxy, not Vision's nose region.
    "outer_lips": (61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291,
                   409, 270, 269, 267, 0, 37, 39, 40, 185),
}
OVAL = (10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
        397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
        172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def adapt_landmarks(landmarks: list) -> dict:
    """Convert image x/right,y/down to the legacy face-box x/right,y/up space.

    The oval-derived box is clipped to the image. Border faces remain observable
    but are excluded from aggregates; no detector confidence is invented.
    """
    if len(landmarks) != 478:
        raise ValueError(f"Unsupported MediaPipe topology: {len(landmarks)} landmarks; expected 478")
    if any(not math.isfinite(v) for p in landmarks for v in (p.x, p.y, p.z)):
        raise ValueError("MediaPipe returned non-finite landmarks")
    oval = [landmarks[i] for i in OVAL]
    x0, x1 = max(0.0, min(p.x for p in oval)), min(1.0, max(p.x for p in oval))
    y0, y1 = max(0.0, min(p.y for p in oval)), min(1.0, max(p.y for p in oval))
    width, height = x1 - x0, y1 - y0
    if width <= 1e-9 or height <= 1e-9:
        raise ValueError("Degenerate MediaPipe face box")

    def region(indices: tuple[int, ...]) -> dict:
        points = [((landmarks[i].x - x0) / width, (y1 - landmarks[i].y) / height) for i in indices]
        raw = b"".join(struct.pack("<dd", *point) for point in points)
        return {"count": len(points), "encoding": "cgpoint-f64le", "data": base64.b64encode(raw).decode("ascii")}

    regions = {name: region(indices) for name, indices in REGIONS.items()}
    # A full forehead-inclusive mesh oval is not Vision's lower face contour.
    regions.update(face_contour=None, left_pupil=None, right_pupil=None)
    return {
        "bounding_box": {"x": x0, "y": 1.0 - y1, "width": width, "height": height},
        "confidence": None,
        "yaw_radians": None, "roll_radians": None, "pitch_radians": None,
        "landmarks": regions,
        "raw_mesh": [[p.x, p.y, p.z] for p in landmarks],
        "quality_flags": ["face-touches-image-border"] if any(
            p.x <= 0 or p.x >= 1 or p.y <= 0 or p.y >= 1 for p in oval
        ) else [],
    }


class MediaPipeBackend:
    """Reuse one CPU IMAGE-mode landmarker for a batch; never download models."""

    def __init__(self, model: Path, max_faces: int = 10,
                 detection_confidence: float = 0.5, presence_confidence: float = 0.5):
        if not model.is_file():
            raise BackendError(f"Model file not found: {model}. Download face_landmarker.task as documented in references/measurement-backends.md")
        try:
            model_bytes = model.read_bytes()
        except OSError as error:
            raise BackendError(f"Cannot read local model: {error}") from error
        try:
            import mediapipe as mp
            import numpy as np
            import PIL
            from PIL import Image, ImageOps
            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision
        except (ImportError, OSError, RuntimeError, ValueError) as error:
            raise BackendError("Cannot load MediaPipe dependencies. Use Python 3.10–3.12 and install requirements-mediapipe.txt; on Linux check native library dependencies. " + str(error)) from error
        self.mp, self.np, self.Image, self.ImageOps = mp, np, Image, ImageOps
        self.max_faces = max_faces
        self.metadata = {
            "name": "mediapipe", "version": mp.__version__,
            "model_sha256": hashlib.sha256(model_bytes).hexdigest(),
            "landmark_schema": "mediapipe-478-rim-centers-alar-proxy-v1",
            "face_box_definition": "clipped-axis-aligned-mesh-oval; not a detector ROI",
            "orientation": "Pillow EXIF transpose; RGB; alpha over white; no mirroring or resizing",
            "pillow_version": PIL.__version__, "numpy_version": np.__version__,
            "running_mode": "IMAGE", "delegate": "CPU", "max_faces": max_faces,
            "min_face_detection_confidence": detection_confidence,
            "min_face_presence_confidence": presence_confidence,
        }
        try:
            options = vision.FaceLandmarkerOptions(
                # A buffer also supports non-ASCII Windows model paths and binds
                # the model hash to the exact bytes used by native inference.
                base_options=python.BaseOptions(model_asset_buffer=model_bytes,
                                                delegate=python.BaseOptions.Delegate.CPU),
                running_mode=vision.RunningMode.IMAGE, num_faces=max_faces,
                min_face_detection_confidence=detection_confidence,
                min_face_presence_confidence=presence_confidence,
                output_face_blendshapes=False, output_facial_transformation_matrixes=False,
            )
            self.landmarker = vision.FaceLandmarker.create_from_options(options)
        except (ValueError, RuntimeError, OSError, AttributeError) as error:
            raise BackendError(f"Cannot initialize Face Landmarker; check model integrity and the pinned package version: {error}") from error

    def extract(self, paths: list[str]) -> list[dict]:
        rows = []
        for path in paths:
            row = {"path": path, "faces": []}
            try:
                data = Path(path).read_bytes()
                row["image_sha256"] = hashlib.sha256(data).hexdigest()
                with self.Image.open(io.BytesIO(data)) as source:
                    if getattr(source, "n_frames", 1) != 1:
                        raise ValueError("Multi-frame image: extract an explicit still frame first")
                    image = self.ImageOps.exif_transpose(source)
                    if "A" in image.getbands() or "transparency" in image.info:
                        rgba = image.convert("RGBA")
                        image = self.Image.new("RGBA", rgba.size, "white")
                        image.alpha_composite(rgba)
                    rgb = self.np.ascontiguousarray(image.convert("RGB"))
                row.update(pixel_width=rgb.shape[1], pixel_height=rgb.shape[0])
                result = self.landmarker.detect(self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb))
                row["faces"] = [adapt_landmarks(face) for face in result.face_landmarks]
                row["detection_limit_reached"] = len(row["faces"]) >= self.max_faces
            except (OSError, ValueError, RuntimeError, self.Image.DecompressionBombError) as error:
                row["error"] = f"Cannot measure image: {error}"
            rows.append(row)
        return rows

    def close(self) -> None:
        self.landmarker.close()
