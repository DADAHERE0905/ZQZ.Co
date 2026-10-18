# Local measurement backends

## Contents

1. Install and provision
2. Measure and review
3. JSON and metric compatibility
4. Identity-profile handoff and shot checks
5. Errors and recovery
6. Verification and remaining work
7. Primary sources

## Install and provision

Run the examples from `capture-face-consistency/`. Use a 64-bit CPython
3.10–3.12 virtual environment. The pinned MediaPipe release has Windows x86-64,
Linux x86-64/glibc 2.28+, and macOS 11+ Intel/universal2 wheels. These are package
availability constraints, not a claim that every OS/architecture has been tested.
Windows ARM, 32-bit Python, Alpine/musl and unsupported Python versions need a
separate environment validation. Use the same pinned backend across the team.

macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-mediapipe.txt
```

Windows PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-mediapipe.txt
```

On Windows replace `python` in subsequent examples with
`.\.venv\Scripts\python.exe` if the environment is not activated. Use
`--output FILE` to avoid PowerShell's version-dependent redirection encoding.
On minimal Linux hosts a native import error may require the distribution's
OpenGL runtime (`libGL.so.1`, e.g. `libgl1` on Debian/Ubuntu). Inference explicitly
uses CPU; no CUDA/GPU or webcam is required.

Provision the official model once, on an approved connected machine if needed:

```bash
python -c "from pathlib import Path; from urllib.request import urlretrieve; Path('models').mkdir(exist_ok=True); urlretrieve('https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task', 'models/face_landmarker.task')"
python -c "import hashlib; from pathlib import Path; print(hashlib.sha256(Path('models/face_landmarker.task').read_bytes()).hexdigest())"
```

The version-1 model used for the local smoke test has SHA-256
`64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff`.
Investigate a mismatch before using the model; do not reuse old calibration.
The measurement command never downloads a model or sends images to a service.
Keep model/likeness/asset rights review in the existing production workflow.
Do not commit private images, raw biometric samples, or model binaries here.
Record `python -m pip freeze`, platform and hardware in the run manifest.

## Measure and review

```bash
python scripts/measure_faces.py hero.png shot02.png --backend mediapipe --model models/face_landmarker.task --raw --output measurements.json
```

`auto` preserves Vision on macOS and selects MediaPipe on other platforms.
There is no automatic fallback after a failure. A mixed-OS project should always
specify `--backend mediapipe`, otherwise its metrics will be incompatible.
The legacy macOS command still works without installing MediaPipe:

```bash
python scripts/measure_faces.py hero.png --backend vision --output vision.json
```

On headless macOS hosts, explicitly try `--vision-cpu-only` if the default
Vision compute path fails. This mode is recorded in the measurement signature;
recalibrate before comparing it with the default mode. Vision errors preserve
the native error description. The macOS CI regression uses this explicit mode.

Inspect every image and the returned `face_box` or `candidate_faces` in the
EXIF-oriented image. Boxes are image-normalized **bottom-left origin** on both
backends. For a top-left pixel overlay use `(x*W, (1-y-height)*H, width*W,
height*H)` with the output pixel dimensions. Raw regional coordinates are
face-box-normalized with y up; `--raw` also emits native MediaPipe image x/y/z
with y down. MediaPipe z is model-relative, not measured metric depth.

Multiple detections fail closed until a human selects `--face-index N` or a
per-image review entry. Indices sort by descending box area, breaking ties by
x/y. Ordering supports review only and is not a tracked identity. Reconfirm it
for each image or frame. The default `--max-faces 10` is a configurable capacity
choice, not a QC threshold; values below 2 are refused. At capacity the image
is blocked even with an index, because additional faces may be hidden. Increase
the capacity or manually crop and review the new image. A detector can still
miss a bystander below capacity: a single detection never proves identity.

After visual review, create a UTF-8 JSON file like this. Copy the **actual**
hashes, signature and box from that run; the placeholders are not runnable:

```json
{
  "schema_version": "1.0",
  "subject_id": "approved-character-id",
  "era_id": "approved-design-state",
  "images": [
    {
      "path": "hero.png",
      "source_id": "S001",
      "image_sha256": "COPY_FROM_IMAGE_ROW",
      "measurement_signature": "COPY_FROM_MEASUREMENT",
      "face_index": 0,
      "face_box": {"x": 0.2, "y": 0.1, "width": 0.6, "height": 0.8},
      "subject_confirmed": true,
      "canonical_approved": false,
      "reviewer": "responsible-reviewer",
      "notes": "Replace this example box and record the visual review"
    }
  ]
}
```

Paths resolve against the command's working directory, including relative paths
in the review file. Use JSON booleans, not strings. Keep one subject and one era
per review file/aggregate. Link source IDs to the rights/source manifest.
Set `canonical_approved=true` only after checking near-frontal pose, neutral or
low expression, no severe pitch, occlusion, beauty filters, overlays or blur.
These are human judgments; this backend does not automatically detect them.
Review entries are audit assertions, not a rights grant or authenticated approval
system. An image/model/preprocessing/selected-box change requires a new review.

```bash
python scripts/measure_faces.py hero.png shot02.png --backend mediapipe --model models/face_landmarker.task --review-file review.json --output reviewed.json
```

Include a review for every frame intended for a canonical aggregate. A bare
`--face-index` selects the subject region but leaves `subject_confirmed=false`.
Every missing review remains outside canonical summaries, even for one face.
The legacy quality boundaries remain ≥160 px high, 96–159 px medium, <96 px low,
using the minimum unrounded face-box pixel dimension. This is a resolution
class only, not a comprehensive face-quality score. `--include-low-quality`
affects the descriptive summary only. Low-resolution or mesh-border faces
cannot enter canonical summaries. MediaPipe's oval box and Vision's detected
box can assign different quality classes to the same image.

## JSON and metric compatibility

Measurement JSON is version `2.0.0`; the identity-profile schema remains `1.0.0`.
The old top-level `status`, `images`, `summary`, normalization/caution/policy
fields and per-image `metrics`/`profile_metrics` are retained. New metadata,
review fields, candidates and canonical summaries are additive. Consumers that
reject unknown keys need an update. Setup failures return a structured
`backend-error`; selection/image failures retain `invalid-selection`.

`measurement_signature` hashes the backend/version/model, task settings,
landmark schema, orientation, image-library versions and preprocessing/code
hash. The runtime platform is recorded separately so the same definition can
be reused across OSes. Source hashes normalize checkout line endings (LF/CRLF).
Equal signatures establish configuration compatibility,
not bit-identical numerical output or calibration validity on new hardware.
Recheck a project holdout after hardware/OS changes. Legacy files without a
signature must be remeasured before automated comparison.

All mapped features are backend-specific. The adapter preserves the historic
unit-square transform `(x/W, y/H)` rather than silently changing old profiles.
It distorts Euclidean distances, aspect ratios and angles when the box is not
square. Horizontal width features use box width; inter-center/brow distances
use unit-square Euclidean distance, and eye/mouth aspects use unit-square
extents. Treat the historical field names as feature names. They are not
anthropometric measurements, true head pose, or pixel-space angles.

| Profile/metric family | MediaPipe definition | Equivalence to Vision |
| --- | --- | --- |
| Eye-center separation; single-eye width; eye height/width | Rim means and extents of official 16-point eye loops; no pupil/iris fallback | Backend-specific, recalibrate |
| Brow-to-eye distance | Mean of the 10 brow points to corresponding rim mean | Backend-specific, recalibrate |
| Nose width and nose/inter-eye ratio | Alar proxy: x extent of points 98 and 327 | Different region definition, recalibrate |
| Mouth widths, height and aspect | Outer 20-point lip loop extents | Backend-specific, expression-dependent |
| Eye-line and mouth-corner tilt | Existing unit-square angle formulas, y up | Diagnostic only, not physical pose |
| `lower_jaw_width_over_contour_width` | `null` | No validated lower-contour mapping |
| `vision_lower_face_contour_height_over_width` | `null` | Full MP oval includes forehead; cannot substitute |
| Raw contour width/height/aspect, lower-jaw width/taper | `null` | Same contour limitation |
| Yaw/pitch/roll | `null` | No validated conversion from effect matrices to Vision's convention |
| FACS, blendshape targets, identity similarity | Not produced | No inferred intensities or identity score |

The exact region indices are versioned in `scripts/mediapipe_backend.py`.
Face size derives from the clipped full mesh oval. Border flags concern the
predicted oval only and cannot prove the face is unoccluded or fully in frame.

## Identity-profile handoff and shot checks

Use only the reviewed `canonical_profile_summary` for a provisional draft
geometry handoff. It automatically adds the four existing `neutral_*` prefixes.
Copy the full `detector` object, `measurement_signature`, normalization and
preprocessing into geometry provenance; retain source/session/era evidence.
Omit unsupported ranges or leave them `null`. Never replace them with zero.
Do not combine people, eras, cameras/poses or expressions merely because a
single summary file accepts multiple inputs.

```python
# After human review, with measurement loaded from reviewed.json:
geometry = profile["geometry"]
geometry["normalization"] = measurement["normalization"]
geometry["detector"] = measurement["detector"]
geometry["measurement_signature"] = measurement["measurement_signature"]
geometry["canonical_near_frontal_ranges"] = {
    key: [stats["min"], stats["max"]]
    for key, stats in measurement["canonical_profile_summary"].items()
}
geometry["source_measurement_notes"].append(
    "Provisional observed ranges only; not calibrated production envelopes."
)
```

Keep `profile_status=draft` or `research-baseline` as appropriate. The script
does not enforce session diversity or authorize a promotion. The existing
three-images/two-sessions visual-stability rule, holdout isolation, rights and
production validator still apply. Observed min/max is not a real holdout 95%
envelope. Run `scripts/validate_profile.py` on the completed profile as before.

For geometry drift diagnostics between reviewed comparable shots:

```bash
python scripts/compare_geometry.py reference-reviewed.json shots-reviewed.json --output geometry-qc.json
```

This requires the same measurement signature, subject and era, a successful
reviewed baseline, and canonical-eligible candidate frames. Without an envelope
it reports only per-field deltas from the baseline median (`diagnostic-only`).
For a range decision, pass `--envelope project-envelope.json` containing:

- `measurement_signature` and `review_scope` copied from reviewed output;
- `calibration_status: "calibrated"`, `owner_approved: true`;
- `calibration_dataset_hash`: actual 64-hex SHA-256 of the project holdout data index;
- nonempty `ranges`: canonical profile-field names mapped to measured `[min,max]`
  boundaries from the project's approved calibration, not this document.

Results use `in-envelope`, `out-of-envelope` or `not-evaluated`; no identity
verdict is generated. A geometry pass covers only the named calibrated fields.
This is useful for an AIGC character bible and shot geometry audit. It does not
replace two independently calibrated identity matchers, an expression extractor,
video tracking, identity-swap detection, temporal QC or the human hero-face review.
Extract still frames explicitly; this adapter runs IMAGE mode without tracking.

When merging into `frame-qc.csv`, leave matcher scores, pose, gaze, expression,
interframe drift and identity-swap fields empty if no corresponding validated
extractor produced them. `geometry_pass` can be true only for `in-envelope`
with the approved calibration recorded; `diagnostic-only`/`not-evaluated` is not
a pass. Keep `frame_pass` undecided until all project gates and human review
are complete; retain the comparison JSON path and signature as evidence.

## Errors and recovery

| Condition | Behavior and recovery |
| --- | --- |
| Missing dependencies, unsupported wheel/native library | Exit 1 with setup error; use the pinned 64-bit environment and inspect stderr |
| No `--model`, missing/corrupt model | Exit 1, `backend-error`; provision/check the local model |
| Vision on Windows/Linux, missing `osascript`, timeout | Exit 1; explicitly select MediaPipe or repair the macOS environment |
| Missing/corrupt/multiframe image, no face, bad selection | Exit 2, per-image error; other images remain measurable |
| Multiple detections without selection, capacity reached | Exit 2 with candidates; manually review/crop or increase capacity |
| Stale review hash/signature/box or conflicting index | Exit 2; perform a fresh review of the current image and configuration |
| Malformed review JSON/types/duplicate path | Exit 1; correct the review file rather than silently ignoring it |
| Incompatible comparison definition/subject/era/envelope | Exit 1; remeasure and recalibrate; do not coerce fields |
| Comparison outside envelope or frame not eligible | Exit 2 with per-frame geometry status; no identity conclusion |
| CLI usage error | Exit 2 via argparse; usage diagnostic goes to stderr |
| Cannot write `--output` | Exit 1; correct the destination/permissions |

Exit 0 means execution/selection succeeded, not that likeness or production QC
passed. Native MediaPipe warnings (including unavailable GPU probes despite CPU
inference) may appear on stderr. JSON stdout/output stays separate. Do not treat
an empty `canonical_profile_summary` as a successful identity profile.

## Verification and remaining work

Fast deterministic tests, with no ML dependencies/model/network:

```bash
python -m unittest discover -s tests -v
```

Opt-in inference tests use local `CFC_MODEL` and `CFC_IMAGE` paths. The image must
be a single-face portrait with sufficient resolution. Public sample data is
execution-test evidence only, not production likeness/rights evidence. For the
exact sample used by CI, download
`https://storage.googleapis.com/mediapipe-assets/portrait.jpg` to a local path.

macOS/Linux:

```bash
CFC_MODEL="$PWD/models/face_landmarker.task" CFC_IMAGE="/path/to/portrait.jpg" python -m unittest discover -s tests -v
```

Windows PowerShell:

```powershell
$env:CFC_MODEL = (Resolve-Path models/face_landmarker.task).Path
$env:CFC_IMAGE = (Resolve-Path C:\path\to\portrait.jpg).Path
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Local verification at authoring used Linux x86-64, Python 3.12.14, MediaPipe
0.10.21, Pillow 11.3.0, NumPy 1.26.4 and the model hash above. The deterministic
suite and real inference exercised geometry/null fields, JSON compatibility,
quality exclusions, manually selected groups/capacity, EXIF/Unicode, missing and
corrupt inputs/models, review binding, provisional profile mapping and diagnostic
comparison. Test review approvals are fabricated software fixtures; they do not
claim that the public portrait passes the canonical-expression or identity gate.

The GitHub Actions matrix runs the same unit/inference tests on Windows, Linux
and macOS, plus the retained Vision entry point on macOS. Treat these as pending
until the actual workflow succeeds. Still requiring authorized production data:
detector recall in ads; profile/occlusion/lighting/makeup/age cases; camera and
compression variation; cross-OS numerical tolerances; identity/error rates;
expression calibration; false acceptance/rejection; temporal drift and swaps.
No commercial acceptance threshold or cross-detector equivalence is claimed.

## Primary sources

- [MediaPipe Python setup and supported platforms](https://ai.google.dev/edge/mediapipe/solutions/setup_python)
- [Face Landmarker Python API, modes, capacity and defaults](https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker/python)
- [Official model bundle and model cards](https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker#models)
- [Pinned 0.10.21 wheel availability](https://pypi.org/project/mediapipe/0.10.21/)
- [Official landmark region connections](https://github.com/google-ai-edge/mediapipe/blob/v0.10.21/mediapipe/python/solutions/face_mesh_connections.py)
