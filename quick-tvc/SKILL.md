---
name: quick-tvc
description: Rapidly create premium original product TVC and social ad video workflows from product images, reference videos, reference images, and scripts. Use when the user wants shot-by-shot advertising direction, keyframe prompts, video generation prompts, edit plans, iteration loops, or review standards for AI-generated commercial videos while preserving product accuracy and avoiding direct copying of reference material.
---

# Quick TVC

## Goal

Turn a product image, reference video, reference images, and script into a polished original ad video plan with prompts, iteration logic, and commercial review standards.

Preserve the product accurately. Extract style from references without copying exact protected shots, sequences, character likenesses, slogans, music, or brand assets that do not belong to the user.

Use the user's language for the final deliverable unless they ask otherwise.

## Intake

Collect or infer:

- Product image: hero product, package, label area, shape, material, color, proportions.
- Reference video: pacing, camera language, lighting, edit rhythm, transition style, product reveal grammar.
- Reference images: art direction, palette, surfaces, props, environment, texture, composition.
- Script: voiceover, overlay text, claims, offer, CTA, required disclaimers.
- Platform: TikTok, Reels, YouTube Shorts, Meta, Amazon, website hero, retail screen, etc.
- Format: aspect ratio, duration, resolution, language, captions, safe zones.
- Brand rules: logo usage, colors, typography, forbidden claims, mandatory warnings.

If format is missing, default to `9:16`, `15-30s`, social-first pacing.

If references are missing, still produce a usable workflow but state which assumptions are being made.

## Reference Analysis

Before writing prompts, summarize the inputs:

1. Product facts: category, physical details, hero angles, must-preserve features.
2. Audience and buying trigger: who the ad should persuade and why.
3. Reference video language: hook style, shot length, camera motion, transitions, lighting, scene density, product reveal.
4. Reference image language: color palette, props, surface, environment, depth of field, material feeling.
5. Script function: hook, problem, benefit, proof, offer, CTA, compliance risk.

Keep this concise. The analysis should guide production, not become a moodboard essay.

## Core Prompt

Use this master instruction when creating the ad plan:

```text
You are a senior commercial director, ad editor, and AI video prompt engineer.

Create a high-quality original product advertisement using:
1. Product image: preserve product shape, packaging, logo placement, material, proportions, and color accurately.
2. Reference video: analyze only the transferable style language, not the exact copyrighted sequence.
3. Reference images: extract mood, lighting, background, props, texture, palette, and composition.
4. Script: convert the message into a clear ad narrative with a strong hook, product demonstration, benefit proof, and CTA.

Output:
- Creative diagnosis of the references
- Ad concept
- Shot-by-shot storyboard
- Keyframe prompts per shot
- Video generation prompts per shot
- Camera movement
- Lighting and art direction
- Edit rhythm
- Text overlay plan
- Sound and music direction
- Review checklist
- Iteration plan

Make the result feel like a premium commercial, not a generic AI video.
Prioritize product accuracy, visual consistency, clean composition, believable motion, strong pacing, and direct-response clarity.
```

## Workflow

### 1. Define Creative Direction

Produce:

- One-sentence concept.
- Visual world.
- Product role: hero object, problem solver, sensory object, lifestyle accessory, transformation trigger.
- Emotional tone.
- Motion language.
- Lighting and color rules.
- Editing rules.

### 2. Build the Shot List

Create a table with 5-10 shots for a 15-30 second ad:

| Time | Job | Visual | Product Role | Camera | Text or VO | Prompt | Risk |
|---|---|---|---|---|---|---|---|

Each shot needs one clear job:

- Hook
- Problem
- Product reveal
- Feature
- Sensory proof
- Lifestyle result
- Offer
- CTA

### 3. Generate Keyframes First

Create still-image or keyframe prompts before full motion. Keyframes protect product accuracy and art direction.

Use this format:

```text
Premium commercial product keyframe for [product].
Scene: [environment].
Composition: [framing and product placement].
Lighting: [lighting style, direction, softness, contrast].
Surface and props: [specific materials and props].
Mood: [emotional tone].
Camera: [lens, angle, depth of field].
Product accuracy: preserve the exact product silhouette, packaging structure, logo area, color, proportions, and material from the product image.
Style reference: inspired by the reference mood and production quality, but using an original composition.
Avoid: warped text, changed logo, extra labels, melted packaging, distorted hands, clutter, low-quality CGI, copied reference framing.
```

### 4. Generate Video Shots

Once keyframes pass review, write per-shot video prompts:

```text
Create a [duration] second premium product ad shot.

Subject:
[product description from product image]

Scene:
[scene description]

Action:
[exact motion, reveal, interaction, pour, splash, turntable, transformation, or lifestyle action]

Camera:
[lens, camera movement, angle, framing, speed]

Lighting:
[lighting direction, softness, contrast, reflections, shadow quality]

Style:
[reference-inspired commercial style, original execution]

Product accuracy:
Preserve the exact product shape, proportions, color, packaging structure, and logo placement from the product image.

Quality:
High-end advertising cinematography, realistic motion, stable product geometry, clean background, crisp focus, controlled reflections, no visual artifacts.

Avoid:
Text distortion, product deformation, fake labels, jitter, melted edges, extra logos, bad hands, duplicated product, unreadable packaging, copied reference sequence.
```

### 5. Assemble the Edit

Specify:

- Total duration.
- Shot order.
- Cut rhythm.
- Transition type.
- Text overlay timing.
- Music direction.
- Sound design moments.
- End card layout.

Keep overlay text short, usually 2-6 words per beat. Put the product and core promise in the first 3 seconds.

## Iteration Loop

Use this loop until the ad reaches publish quality:

```text
Analyze references -> Create concept -> Build shot list -> Generate keyframes -> Review keyframes -> Generate video shots -> Review shots -> Replace weak shots -> Assemble edit -> Review final ad -> Polish -> Export
```

For each iteration, report:

- What failed.
- Why it failed.
- Exact fix.
- Revised prompt.
- Whether to regenerate, edit, upscale, color grade, stabilize, or replace.

Change one layer at a time:

1. Product accuracy
2. Composition
3. Motion
4. Lighting
5. Edit rhythm
6. Text clarity
7. Final polish

Do not regenerate the whole ad if one shot fails. Patch the weakest shot first.

## Review Standard

Score each category from 1-5.

### Product Accuracy

- 5: Product matches the source image closely.
- 3: Product is recognizable with minor differences.
- 1: Shape, logo area, color, packaging, or proportions changed.

Fail immediately if the product is materially incorrect.

### Commercial Quality

- 5: Looks like a real premium ad.
- 3: Good but visibly generic or AI-made.
- 1: Amateur, messy, unclear, or cheap-looking.

### Reference Alignment

- 5: Captures mood, pacing, camera, and quality level without copying.
- 3: Some stylistic similarity.
- 1: Unrelated to references.

### Motion Quality

- 5: Smooth, intentional, physically believable.
- 3: Usable with small artifacts.
- 1: Jitter, warping, unstable product, broken physics.

### Composition

- 5: Clear focal point, clean negative space, strong product hierarchy.
- 3: Acceptable but not premium.
- 1: Cluttered, awkward, badly cropped, or visually confusing.

### Lighting and Material

- 5: Product texture and reflections feel controlled and realistic.
- 3: Lighting works but lacks polish.
- 1: Flat, muddy, overexposed, or inconsistent.

### Message Clarity

- 5: Hook, benefit, and CTA are instantly clear.
- 3: Understandable but not sharp.
- 1: Viewer cannot tell what is being sold or why.

### Platform Fit

- 5: Fits aspect ratio, pacing, safe zones, subtitles, and attention span.
- 3: Needs minor formatting changes.
- 1: Wrong format or weak first 3 seconds.

## Pass Criteria

The video plan or generated ad is ready only when:

- Product accuracy is at least 4/5.
- Commercial quality is at least 4/5.
- Motion quality is at least 4/5.
- Message clarity is at least 4/5.
- No shot contains distracting AI artifacts.
- The first 3 seconds clearly hook the viewer.
- The final CTA is readable and platform-safe.
- Reference influence is stylistic, not a direct copy.

## Final Output

Return:

1. Creative direction
2. Reference style analysis
3. Shot-by-shot storyboard
4. Keyframe prompts
5. Video prompts
6. Edit plan
7. Review checklist
8. Iteration notes
9. Final export recommendations

When the user provides actual outputs to review, switch to review mode: score the work, name the top problems, and produce revised prompts or edit instructions for the next loop.
