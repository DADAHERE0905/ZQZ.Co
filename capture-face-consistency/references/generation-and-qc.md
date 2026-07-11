# Generation and consistency QC

Use this reference when selecting an identity method, building a run manifest, diagnosing drift, or setting acceptance gates.

## Contents

1. Core architecture
2. Method selection
3. Still-image workflow
4. Video workflow
5. Common failures
6. Matcher calibration
7. Per-frame and shot metrics
8. Commercial license audit
9. Primary sources

## Core architecture

Use four independent layers:

1. **Identity passport** — pose/era-clustered embeddings, geometry ranges, texture cues, medoids.
2. **Expression library** — FACS AU or blendshape centroids, tolerances, head pose, gaze, mouth state.
3. **Generation recipe** — exact models, hashes, preprocessing, weights, controls, sampler, seed, precision, hardware.
4. **QC** — two independent identity matchers, geometry, expression, temporal checks, and human review.

Face-recognition embeddings answer “does this resemble the subject?” They intentionally suppress much expression variation and cannot define a signature expression. Keep identity and expression scores separate.

## Method selection

| Method | Data/training | Strength | Main failure | Best role |
|---|---|---|---|---|
| Reference embedding | One image runs; multiple views preferred | Fast, measurable | Inherits pose/expression; misses fine identity cues | Baseline conditioning and QC |
| InstantID | Single image, no subject fine-tune | Strong zero-shot identity with face embedding and landmarks | High weight reduces text/expression control; official checkpoint chain is research-only | Fast look development |
| IP-Adapter FaceID | Single or multiple images depending variant | Mature ecosystem and composability | FaceID/CLIP strength trade-off; can copy reference styling | Composable still pipeline |
| PuLID | Single reference, no subject fine-tune | Good identity/editability balance; SDXL and FLUX variants | Male fidelity can vary by version; strong ID loss can reduce natural detail | First zero-shot still benchmark |
| PhotoMaker | Multiple references, no per-subject training | Stacked multi-image ID can cover views | Inconsistent references produce an average face | Multi-view rapid baseline |
| Character LoRA | A proposed starting range is 15–30 curated production images; coverage matters more than count | Compact, reusable long-term prior | Underfit looks generic; overfit binds clothes/background/expression | Recurring production character |
| DreamBooth / DreamBooth-LoRA | Few images can work; production needs diversity | Strong subject binding | Overfit, language drift, training cost | High-value long-term identity |
| LivePortrait | Source portrait plus driving video | Controllable talking/performance portrait | Depends on source/driver quality and license | Interviews and close talking shots |
| ConsisID | Identity-conditioned text-to-video | Free-scene identity-preserving video | Heavy compute; environment affects repeatability | Short free-scene video tests |

Benchmark candidates on one fixed matrix. The following is an example planning proposal, not a universal or measured recipe:

- a proposed 20–40 prompts, adjusted to the shot list and risk;
- identical target era;
- front, ±15°, ±30°, profile, high/low camera, close/medium/full shot;
- neutral, signature expression, speech, and one high-intensity expression;
- occlusion and re-entry cases for video;
- same crop, pose/depth controls, seeds, and output size where supported.

Compare pass rate, p05, and worst failures. Do not select by a hand-picked best image.

## Precision discipline

- Do not provide universal adapter weights, LoRA ranks, learning rates, step counts, crop coordinates, repair-window sizes, seeds, or AU intensities.
- Use exact values only when they come from measured subject artifacts, current official implementation guidance, or a completed controlled sweep. Cite or record that provenance.
- For unknown values, write `null` and provide a bounded experiment matrix whose factors are the chosen implementation's documented controls.
- For one-view fictional characters, state that unobserved profile, skull, ear, and nose-projection geometry must be authored and approved as canon.
- Treat FACS/blendshape targets as subject-specific only after multiple approved examples are measured. Generic expression examples may define a capture brief, not the character's exact parameters.
- Treat the accepted false-match rate as a project risk decision. Do not silently default to `0.001` or any other operating point.

## Still-image workflow

### Rapid look development

1. Approve one high-quality hero reference per pose cluster.
2. Hold pose/depth and prompt constant.
3. Sweep identity strength from low to high.
4. Stop increasing strength when expression, age state, text adherence, or skin naturalness begins to freeze.
5. Save every model, preprocessing, and weight value in `run-manifest.json`.

### Long-term recurring character

1. Train the LoRA on the selected era only.
2. Balance angles, expressions, wardrobe, and backgrounds; down-weight bursts.
3. Caption variable elements so the identity token does not absorb them.
4. Test on unseen holdout prompts and real holdout poses.
5. Add a low-to-medium ID adapter only for shots that need extra facial fidelity.
6. Use pose/depth/landmark controls instead of raising identity strength to fix composition.
7. Keep expression in a separate AU/blendshape/driver channel.

Do not max out LoRA and adapter simultaneously. Typical symptoms are pasted-on face, frozen expression, reference-pose leakage, and weak text control.

## Video workflow

### Talking portrait or interview

- Prefer a source-portrait plus approved driving performance route when free-scene generation is unnecessary.
- Validate the source image first.
- Preserve blink, lip closure, gaze, head-pose range, and signature expression separately.
- Track the face region and use temporally coherent local repair; avoid unrelated per-frame face swaps.

### Free-scene video

- As a planning proposal, start experiments with 3–6 second shots and revise from measured drift and compute limits.
- Approve the hero first frame before image-to-video or identity-conditioned generation.
- Carry an approved keyframe or previous approved terminal frame into the next shot.
- Revalidate after profile turns, hand/prop occlusion, leaving/re-entering frame, lighting cuts, and camera-distance changes.
- Save model hashes, exact dependencies, precision, hardware, preprocessing, and all random states. ConsisID's official repository warns that the same seed and prompt can differ across machines.

### Multi-person scenes

- Assign explicit tracked subject IDs and face boxes.
- Condition one subject at a time where the model cannot guarantee separation.
- Count any identity exchange as a hard failure.

## Common failures and responses

| Failure | Likely cause | Response |
|---|---|---|
| Reference pose/expression copied | Single-image pollution or excessive image conditioning | Add multi-view medoids; reduce adapter; separate expression control |
| “Almost him” generic face | Weak identity prior or poor crop | Improve authorized gallery; retrain LoRA; correct crop/preprocess |
| Pasted-on face | ID/LoRA weights too high | Reduce identity strength; restore structure/text controls; regenerate |
| Expression frozen | Smile or expression absorbed into identity | Rebuild training captions/data; move expression to AU/driver channel |
| Small face collapses | Too few face pixels | Generate closer or higher resolution; use approved keyframe; repair coherently |
| Profile drift | Missing profile gallery | Acquire standardized left/right profile references; do not guess |
| Re-entry becomes another person | Video identity state lost | Track ID; use approved re-entry keyframe; shorten shot; reject failed segment |
| Stable but wrong identity | Interframe score compares only adjacent generated frames | Compare every frame to real pose-clustered prototypes |
| High score but human likeness weak | Matcher reward hacking or crop bias | Use an independent second matcher, geometry, expression, and blinded human review |
| LoRA binds clothes/background | Dataset/caption imbalance | Diversify and recaption; retrain; verify prompt disentanglement |
| Multi-person face swap | Conditioning cross-talk | Explicit face boxes/IDs; separate passes; reject all swaps |

## Matcher calibration

For matcher `m`, build genuine and impostor score distributions using the exact production crop and preprocessing.

Choose an allowed false-match rate `alpha` and set:

```text
tau_m = quantile(impostor_scores, 1 - alpha)
```

Report false non-match rate on genuine holdout at `tau_m`. If the distributions overlap heavily, improve data or change the matcher; do not simply lower the threshold.

Have the accountable project owner choose `alpha` from legal, reputational, and operational risk. Confirm that the number of impostor comparisons is sufficient to estimate that operating point; otherwise report the statistical limitation and collect more calibration data.

For pose cluster `k`:

```text
p_k = normalize(sum(e_i))
s_t = max_k cosine(e_t, p_k)
```

Calibrate the exact same clustering and `max` rule. Save medoids so a reviewer can inspect the real prototype images.

Use two independent matchers for final acceptance. Avoid optimizing and accepting with the same encoder.

## Metrics

### Per frame

- face detected and face-pixel size;
- selected tracked subject ID;
- matcher A and B scores plus margins;
- closest pose-cluster ID;
- head yaw, pitch, roll;
- gaze error;
- canonical geometry deviation;
- expression AU/blendshape weighted error;
- interframe embedding drift;
- occlusion and blur flags;
- pass/fail and reason.

### Per shot

- `face_detection_rate`;
- `median_face_sim`, `p05_face_sim`, `min_face_sim` for each matcher;
- `frame_pass_rate`;
- `max_consecutive_fail_frames`;
- `hard_negative_margin`;
- `identity_swap_count`;
- geometry and expression pass rates;
- temporal flicker and motion smoothness;
- human-review decision.

### Illustrative operational gate proposal

- Hero close-up: face detected and both matchers pass every frame.
- Ordinary shot: detection rate ≥99%; dual-matcher pass rate ≥95%; no two consecutive failed frames.
- `p05` above each calibrated threshold, not only a high mean.
- Zero identity swaps.
- Geometry and target expression within the real holdout 95% envelope.
- Human review required.

These are not measured identity thresholds or universal requirements. Record them as `proposed`, have the accountable owner approve or replace them, and tighten them for legal, reputational, or hero-asset risk.

## Commercial license audit

Audit every dependency independently:

- subject and likeness rights;
- source image/video copyright and AI/ML permission;
- face detector and recognition weights;
- ID adapter code and checkpoint;
- base image/video model;
- character LoRA training data and output terms;
- driving video or motion-capture rights;
- intended advertising/endorsement context.

[InsightFace](https://github.com/deepinsight/insightface) states that its code is MIT, but its provided training data and models trained on it are non-commercial research by default; separate licensing is offered for face-recognition and swap models. [InstantID](https://github.com/instantX-research/InstantID) states that its code is Apache-2.0 while its released checkpoints and the required InsightFace face models are research-only. Therefore, repository code licenses do not settle checkpoint or production-chain rights.

## Primary sources

- [InstantID paper](https://arxiv.org/abs/2401.07519) and [official repository](https://github.com/instantX-research/InstantID)
- [IP-Adapter official repository](https://github.com/tencent-ailab/IP-Adapter) and [FaceID technical page](https://github.com/tencent-ailab/IP-Adapter/wiki/IP%E2%80%90Adapter%E2%80%90Face)
- [PuLID paper](https://arxiv.org/abs/2404.16022) and [official repository](https://github.com/ToTheBeginning/PuLID)
- [PhotoMaker CVPR paper](https://openaccess.thecvf.com/content/CVPR2024/papers/Li_PhotoMaker_Customizing_Realistic_Human_Photos_via_Stacked_ID_Embedding_CVPR_2024_paper.pdf) and [official repository](https://github.com/TencentARC/PhotoMaker)
- [DreamBooth CVPR paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Ruiz_DreamBooth_Fine_Tuning_Text-to-Image_Diffusion_Models_for_Subject-Driven_Generation_CVPR_2023_paper.pdf)
- [LoRA official repository](https://github.com/microsoft/LoRA)
- [ConsisID CVPR paper](https://openaccess.thecvf.com/content/CVPR2025/papers/Yuan_Identity-Preserving_Text-to-Video_Generation_by_Frequency_Decomposition_CVPR_2025_paper.pdf) and [official repository](https://github.com/PKU-YuanGroup/ConsisID)
- [LivePortrait official repository](https://github.com/KlingAIResearch/LivePortrait)
- [OpenFace official repository](https://github.com/TadasBaltrusaitis/OpenFace)
- [MediaPipe Face Landmarker blendshapes](https://ai.google.dev/edge/api/mediapipe/python/mp/tasks/vision/drawing_styles/face_landmarker/Blendshapes)
- [ArcFace CVPR paper](https://openaccess.thecvf.com/content_CVPR_2019/html/Deng_ArcFace_Additive_Angular_Margin_Loss_for_Deep_Face_Recognition_CVPR_2019_paper.html)
- [NIST face-image quality evaluation](https://pages.nist.gov/frvt/html/frvt_quality.html)
- [VBench official repository](https://github.com/Vchitect/VBench)
