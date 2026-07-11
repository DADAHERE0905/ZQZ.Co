---
name: capture-face-consistency
description: Build a source-backed face bible and repeatable facial-identity consistency workflow for a manually confirmed real person or fictional character. Use when Codex must research or curate reference images, capture stable facial geometry and distinctive expressions, define FACS/blendshape prototypes, prepare identity-profile and run manifests, choose reference adapters or character LoRA/DreamBooth methods, control a face across generated images or video, diagnose identity drift, or establish quantitative and human-review QC. Supports authorized production assets and public-web research indexes while keeping rights, age states, identity, expression, styling, and generation settings separate.
---

# Capture Face Consistency

Treat consistency as a versioned system:

`identity passport + expression library + generation recipe + calibrated QC`

Do not promise permanent identity lock from a prompt, seed, single image, or single face-similarity score.

## Enforce boundaries

- For a real person, separate a public research index from production-authorized assets. Do not treat an online image, editorial license, or open-source code license as permission for AI training or commercial likeness use.
- Record provenance and rights for every source. Never remove watermarks or route around license restrictions.
- Analyze only the manually confirmed subject. Do not use this workflow to identify unknown people. Treat every group image and same-name result as unverified until a human confirms the face box.
- Keep minors, family members, hosts, teammates, and bystanders out of the subject galleries unless separately in scope and authorized.
- Keep stable identity, age state, expression, styling, lighting, and pose as separate fields. Do not encode an expression or hairstyle as identity.
- State gaps. Do not claim the profile view, current age state, or an expression is locked when the corresponding references are missing.
- Do not invent exact geometry, FACS/blendshape intensities, crop ratios, adapter weights, LoRA hyperparameters, seeds, or QC thresholds. Use measured values from verified artifacts or implementation-specific values from current primary documentation and label their provenance. Otherwise write `null`, define the acquisition need, and propose a controlled sweep.
- Preserve the schema's exact enums and field meanings. In particular, use only `draft`, `research-baseline`, or `production-ready` for `profile_status`; put blockers or qualifiers in separate fields instead of inventing a descriptive status.
- Label acquisition counts, benchmark matrices, shot lengths, and operating gates as `proposed` until project evidence or an owner-approved policy makes them binding. Never present planning defaults as measured subject parameters or universal recipes.
- For a fictional character with unseen views, require a character designer to approve or author those views. Artist-approved synthetic or sculpted views can establish canon, but they are not facts recovered from one frontal portrait.

## Produce the core artifacts

Create these artifacts for a complete run:

1. `source-manifest.csv` — provenance, scene, era, angle, expression, rights, production eligibility, verification notes.
2. `identity-profile.json` — stable features, era states, normalized geometry, expression prototypes, gallery plan, known gaps.
3. `run-manifest.json` — identity-profile and source-manifest hashes, exact prompt content or resolvable prompt files, model/checkpoint hashes, preprocess version, crops, adapters, LoRA weights, sampler, seed, precision, hardware, and environment.
4. `frame-qc.csv` — per-frame detection, two matcher scores, pose, geometry, expression error, drift, pass/fail reason.
5. A concise human report — representative features, signature expressions, recommended stack, rights blockers, and next capture requirements.
6. A failure contact sheet when generated media is evaluated.

Copy the starting files from `assets/`. Read `references/identity-profile-schema.md` before writing the profile. Read `references/generation-and-qc.md` before recommending a model stack or thresholds. Read `references/source-and-rights.md` for every real-person task.

Validate every handoff:

```bash
python3 "$CODEX_HOME/skills/capture-face-consistency/scripts/validate_profile.py" \
  identity-profile.json --manifest source-manifest.csv
```

The validator intentionally applies much stricter rights, gallery-isolation, local-file, matcher-independence, and calibration checks when `profile_status` is `production-ready`.

## Follow the workflow

### 1. Fix the subject and target state

- Record a disambiguated subject ID and evidence that the references depict the intended subject.
- Choose the target era or age state before curating. Build separate profiles when a project needs materially different ages.
- Record intended medium: still image, free-scene video, talking portrait, performance capture, or all of them.
- Record intended use: research, internal look development, or commercial production.

### 2. Build a source and rights manifest

- Search official pages, brand campaigns, federation/event archives, licensed photo agencies, broadcaster interviews, and subject-provided material.
- Prefer original pages and masters over reposts. Record both page URL and direct asset URL only when appropriate.
- Mark each item as `research-index-only`, `production-authorized`, `conditional`, or `rejected`.
- Confirm every candidate visually. Reject wrong people, host-only frames, mirrored duplicates, AI-generated images, extreme beauty filters, and near-identical burst frames.
- Treat group photos as manual-crop sources. Never let “largest detected face” silently choose the subject.

Require at least three verified images from at least two independent sessions before calling a visual feature stable.

### 3. Split galleries before measuring or training

Use non-overlapping roles. The counts below are acquisition-planning proposals, not universal minimums; adjust them after deduplication, coverage review, rights review, and pilot results:

- `conditioning_gallery`: 8–15 authorized images across front, ±15°, ±30°, both profiles, and varied light.
- `expression_gallery`: 3–5 examples per approved expression with AU/blendshape, head-pose, gaze, and mouth-state labels.
- `training_set`: normally 15–30 diverse authorized images for a reusable character LoRA.
- `holdout_set`: 10–20 real images never used for training, conditioning, model selection, or prompt tuning.
- `hard_negatives`: matcher calibration only; never train them into the subject.

Cluster identity references by `era × yaw band × appearance state`. Save a normalized embedding centroid and a medoid for each cluster; do not rely on one global average vector.

### 4. Extract geometry with quality gates

On macOS, run:

```bash
python3 scripts/measure_faces.py IMAGE [IMAGE ...]
```

For a single detected face, no index is needed. When multiple faces are detected, the script fails closed until a human visually confirms the subject and reruns with `--face-index N`. Add `--raw` when landmark coordinates are required.

The script uses local macOS Vision landmarks and does not identify people. It labels detected faces as:

- high quality: minimum detected face dimension ≥160 px;
- medium quality: 96–159 px;
- low quality: <96 px, excluded from aggregate summaries by default.

On non-macOS systems, use a local Face Landmarker such as MediaPipe and map the results to the same normalized fields. If no validated landmark tool exists, write qualitative features and leave numeric fields null; never invent ratios.

Measure canonical geometry only on manually confirmed, near-frontal, neutral or low-intensity frames. Exclude crying, exertion, open-mouth victory, heavy occlusion, text overlays, severe pitch, and strong profile views from the canonical aggregate.

### 5. Separate stable identity from dynamic expression

For each facial region, record:

- qualitative anchor and negative anchor;
- confidence (`high`, `medium`, or `low`);
- supporting source IDs across sessions and eras;
- normalized geometry range where measurement is valid;
- age-state adjustments.

Create expression prototypes independently. Use FACS AU or blendshape centroids and tolerances plus plain-language cues. Include at minimum:

- neutral or attentive;
- low-intensity social smile;
- subject-signature expression;
- one high-intensity emotion when relevant;
- speech or performance expression when video is in scope.

When AU/blendshape values have not been measured by a validated extractor or qualified coder, store candidate units in `active_aus`, set numeric target fields to `null`, and record only separately labeled observed landmark samples. A single observed example is evidence, not a production target or tolerance band.

Do not infer personality, health, ethnicity, or intent from facial appearance. Describe visible geometry and movement only.

### 6. Build a prompt anchor, but do not depend on it

Write three blocks:

1. immutable identity geometry;
2. one variable expression prototype;
3. negative drift constraints.

Keep styling, wardrobe, hair, background, camera, and light outside the immutable identity block unless the project explicitly defines them as character canon.

### 7. Select the generation stack by production horizon

- For rapid zero-training look development, benchmark several reference-conditioned methods on the same prompt/seed/pose matrix. Compare success rate and the worst 5%, not the best sample.
- For a recurring high-value character, train a character LoRA or DreamBooth-LoRA as the long-term identity prior, then add only the ID-adapter strength a shot needs.
- Use pose, depth, and landmark controls for composition. Do not increase ID strength to solve a pose problem.
- Drive expressions with AU/blendshape targets or approved driving footage. Do not train a permanent smile into the identity token.
- For video, approve the hero first frame, work in short shots, preserve a tracked face ID, and revalidate after occlusion or re-entry.

Do not prescribe universal ranks, learning rates, step counts, adapter strengths, crop coordinates, or video repair windows. Start from the chosen implementation's current official guidance, define a small experiment matrix, and select from holdout results.

Audit the entire license chain: source images, face encoder, adapter checkpoint, base model, LoRA training data, video model, and output terms. Code under MIT or Apache does not make bundled pretrained models commercially usable.

### 8. Calibrate QC on the project

Never copy a cosine threshold from another paper or project. For each independent matcher:

1. build genuine pairs from real holdout images;
2. build impostor pairs from hard negatives and ordinary negatives;
3. have the project owner choose an accepted false-match rate based on risk, then set the threshold;
4. report false non-match rate at that threshold;
5. preserve the exact crop and preprocessing hash.

Require two independent matchers for production hero faces. Use identity scores for “same person” and a separate AU/blendshape or landmark test for “correct signature expression.”

Report at least detection rate, median/p05/min identity score, frame pass rate, longest failure run, hard-negative margin, interframe drift, geometry deviation, expression error, pose/gaze error, and identity swaps.

Prefer real holdout 95% envelopes for geometry and expression. Keep human review mandatory for likeness, age state, signature expression, artifacts, and unintended lookalike drift.

Do not default to a particular false-match rate, hard-negative count, or cosine threshold. Report what sample size is available and whether it can support the requested operating point.

### 9. Stop and report blockers

Do not advance to production generation when any of these is true:

- likeness or AI-training rights are unknown;
- the target era lacks authorized images;
- no standard profile references exist for profile-heavy shots;
- group images have not been manually boxed;
- holdout images leaked into training or prompt/model selection;
- the matcher threshold is uncalibrated;
- a single encoder is used for both optimization and final acceptance;
- model or checkpoint license is research-only for a commercial job;
- the full run environment cannot be reproduced.

Return the exact missing authority, asset, or validation step instead of weakening the gate.
