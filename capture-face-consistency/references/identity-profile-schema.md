# Identity profile schema

Use this reference when creating or revising `identity-profile.json`.

## Contents

1. Required top-level fields
2. Era states
3. Stable feature records
4. Geometry and quality gates
5. Expression prototypes
6. Gallery plan
7. Generation and QC policy

## Required top-level fields

```json
{
  "schema_version": "1.0.0",
  "profile_status": "draft|research-baseline|production-ready",
  "subject": {},
  "rights": {},
  "era_states": [],
  "stable_identity_features": {},
  "geometry": {},
  "expression_prototypes": [],
  "prompt_anchor": {},
  "gallery_plan": {},
  "generation_recipe_policy": {},
  "qc": {},
  "known_gaps": []
}
```

Set `production-ready` only when the production galleries are authorized, target-era coverage is complete, holdout material is isolated, and matcher/expression gates are calibrated.

These are exact enum values. Do not append blockers or descriptive text to `profile_status`; store those in `known_gaps`, gallery status fields, or a dedicated blocker field.

## Subject and rights

Record:

- a stable, disambiguated `subject_id`;
- display name and identifying context;
- whether the subject is a real person or fictional character;
- default target era;
- public-index versus production-asset distinction;
- likeness, AI/ML training, derivative-work, advertising, territory, and term status;
- source-manifest path;
- unresolved model-license dependencies.

For `production-ready`, store a `production_authorized_scope` object covering likeness, AI training when used, reference conditioning, biometric processing for matcher/QC use, derivative outputs, commercial advertising, territory, and term. The first six permission fields are booleans and must be exactly `true` when applicable; strings such as `"approved"` or `"DENIED"` are not accepted as substitutes. The validator also requires the per-asset production rights fields in the manifest.

Do not store sensitive personal data that is unnecessary for visual production.

## Era states

Use one era state for every materially different facial age or canonical design state.

```json
{
  "era_id": "current-2026",
  "age_range_approx": "40-plus",
  "shape_notes": ["fuller lower face"],
  "texture_notes": ["natural eye-area maturity"],
  "source_ids": ["S101", "S102"]
}
```

Never create one global identity average across childhood, adolescence, adulthood, makeup/prosthetic states, or intentionally redesigned fictional versions.

## Stable feature records

For each region, store:

```json
{
  "value": "plain-language visible geometry",
  "confidence": "high|medium|low",
  "positive_anchors": ["what must remain"],
  "negative_anchors": ["common drift to reject"],
  "source_ids": ["S001", "S004"]
}
```

Recommended regions:

- overall face shape;
- forehead/hairline only when stable and visible;
- cheeks and zygomatic volume;
- jaw and chin;
- eyebrows;
- eyes and eyelids;
- nose;
- philtrum and nasolabial region;
- mouth and lips;
- ears only when they materially aid profile consistency;
- stable scars, tattoos, or fictional markings only when authorized and consistently visible;
- facial hair as either identity canon or styling variable.

Do not infer personality, intelligence, health, ancestry, or intent from appearance.

## Geometry

Normalize landmark coordinates and distances to the detected face bounding box unless a validated 3D coordinate system is used. Store the detector, version, preprocessing hash, and orientation handling.

Recommended 2D metrics:

| Field | Definition |
|---|---|
| `eye_center_distance_over_face_width` | Euclidean distance between pupil or eye-region centers |
| `single_eye_width_over_face_width` | Horizontal landmark extent for each eye |
| `eye_height_over_eye_width` | Vertical eye extent divided by horizontal extent |
| `brow_eye_center_distance_over_face_width` | Brow center to pupil/eye center distance |
| `nose_width_over_face_width` | Horizontal extent of nose landmarks |
| `nose_width_over_eye_center_distance` | Nose width divided by inter-eye center distance |
| `mouth_width_over_face_width` | Horizontal extent of outer-lip landmarks |
| `mouth_width_over_eye_center_distance` | Mouth width divided by inter-eye center distance |
| `mouth_height_over_mouth_width` | Outer-lip vertical extent divided by width |
| `lower_jaw_width_over_contour_width` | Width of the lower 38% face-contour points divided by full contour width |
| `vision_lower_face_contour_height_over_width` | Landmark face-contour height divided by width; detector-specific and forehead-excluding |

Treat detector-specific metrics as internal comparison features, not anthropometric truth.

`scripts/measure_faces.py` emits a `profile_metrics` object using the field names in this table. After a frame passes the neutral/low-expression canonical gate, map `eye_height_over_eye_width`, `mouth_width_over_face_width`, `mouth_width_over_eye_center_distance`, and `mouth_height_over_mouth_width` into the profile's corresponding `neutral_*` range fields. Never add the `neutral_` label to a smile, speech, crying, shout, or exertion sample.

### Quality classes

- high: minimum detected face dimension ≥160 px;
- medium: 96–159 px;
- low: <96 px.

Exclude low-quality faces from canonical aggregates. Require manual confirmation, near-frontal pose, no severe pitch, and neutral or low-intensity expression.

Store ranges as arrays:

```json
"eye_center_distance_over_face_width": [0.35, 0.39]
```

Prefer the real authorized holdout's 95% envelope. If an initial public-research range is used, label it as provisional and recalibrate before production.

## Expression prototypes

Keep expression outside stable identity.

Choose one priority representation for the whole profile (for example, all strings such as `primary`, `secondary`, and `special-shot`, or all numeric ranks). Do not mix strings and numbers.

```json
{
  "expression_id": "signature-smile",
  "priority": "primary",
  "evidence_status": "visual-cross-check-only; production-target-unset",
  "active_aus": ["AU6", "AU12"],
  "facs_0_to_1": null,
  "facs_measurement_status": "not-measured-with-validated-au-extractor",
  "blendshape_targets": null,
  "landmark_targets": null,
  "landmark_observed_samples": {},
  "head_pose_range_degrees": null,
  "gaze": "camera|off-camera|variable",
  "mouth_state": "closed|teeth-visible|open|speech",
  "visual_cues": ["plain-language checks"],
  "anti_cues": ["nearby but wrong expression"],
  "source_ids": ["S020", "S021"]
}
```

Record a centroid and tolerance from multiple examples when a validated AU/blendshape extractor is available. Do not assign false precision from visual inspection. Visual inspection may identify candidate AU units, but numeric intensities remain `null`. Keep observed 2D samples separate from production targets, and label the source IDs and extractor provenance.

## Gallery plan

Keep these roles non-overlapping:

- conditioning;
- expression;
- training;
- holdout;
- hard negatives.

For each gallery, store source IDs, file hashes, perceptual hashes, crop hashes, license status, era, pose, expression, occlusion, face pixel size, and acceptance reason.

The recommended representation is a gallery-level `source_ids` list plus `member_records` for crop- or role-specific facts. Source-wide provenance and rights may be joined from `source-manifest.csv`; do not duplicate conflicting values. Before `production-ready`, set `leakage_audit_passed=true`, mark target-era coverage complete for the approved shot list, and keep conditioning, expression, training, holdout, and hard-negative IDs mutually exclusive.

Store pose-aware identity clusters separately:

```json
{
  "cluster_id": "current-front-neutral",
  "era_id": "current",
  "yaw_band": "front",
  "appearance_state": "neutral",
  "source_ids": ["S101", "S102"],
  "embedding_model_hash": "...",
  "preprocessing_hash": "...",
  "centroid_file": "...",
  "centroid_sha256": "...",
  "medoid_source_id": "S101"
}
```

Place these objects in `gallery_plan.identity_clusters`; never serialize raw biometric embeddings into a public research package without the required permission and access controls.

If a count or coverage matrix is still a planning recommendation, name or label it as `proposed`; do not imply that it was measured or is mandatory for every subject.

## Prompt anchor

Store:

- immutable visible identity block;
- negative drift block;
- warning that prompt text is auxiliary;
- styling and expression fields outside identity.

Avoid using the subject's name as the only identity instruction.

## QC policy

Include:

- matcher names, versions, model hashes, preprocessing hashes;
- calibrated thresholds and target false-match rates;
- genuine/impostor sample counts;
- geometry and expression envelopes;
- per-shot gates;
- human-review roles;
- known limitations and missing views.

For `production-ready`, each matcher must have a unique `matcher_id`, `model_family`, model SHA-256, and preprocessing SHA-256. Each calibrated threshold must identify its matcher, numeric threshold, owner-approved false-match operating point, and calibration-data hash. Also record positive genuine/impostor counts, holdout isolation, and calibrated geometry/expression envelope hashes. The validator rejects duplicate matcher families, gallery leakage, research-only assets, missing local production files, or incomplete production rights.

Use `assets/identity-profile.template.json` as the starting artifact.
