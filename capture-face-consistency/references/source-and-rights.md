# Source, verification, and rights workflow

Use this reference for every real-person task and for copyrighted fictional-character references.

## Contents

1. Two-library rule
2. Source priority
3. Manual verification
4. Rights fields
5. Rejection rules
6. Production gate

## Two-library rule

Maintain two physically and logically separate libraries:

### Public research index

Use for discovery, visual comparison, gap analysis, and deciding what to license or reshoot. Store links, thumbnails only when permitted, provenance, visible features, and rights notes. Do not silently promote these files to training or conditioning.

### Production-authorized library

Use only assets whose rights cover the intended operation. Record the contract or license reference and scope. Keep a reproducible hash inventory.

Never mix the folders. A public URL is not authorization.

## Source priority

Prefer sources in this order:

1. subject/rightsholder-provided masters with explicit AI/ML and derivative rights;
2. commissioned standardized capture;
3. brand or production masters with written permission;
4. appropriately licensed photo/video masters;
5. official federation, broadcaster, publisher, or event archives for research and cross-checking;
6. reputable editorial agencies for research and licensing inquiry;
7. reposts only as leads back to an original source.

Reject unsourced social reposts, fan edits, beauty-filtered composites, unknown AI images, and search-engine thumbnails as production assets.

## Manual verification

For every candidate:

- confirm the page title, date, event, and named subject;
- visually confirm the correct face, especially in group images;
- check for same-name people, misspellings, nearby celebrities, hosts, teammates, relatives, and children;
- note mirroring, crops, overlays, watermarks, lens distortion, beauty filters, and compression;
- identify duplicate or burst-sequence frames with perceptual hashes;
- record whether the image is original, a screenshot, a repost, or a licensed agency preview;
- reject model-generated or heavily altered images from the ground-truth identity galleries.

Never default to the largest detected face in a group image. Save an explicit subject bounding box or crop after human confirmation.

## Rights fields

Record at least:

- `rights_status`: `research-index-only`, `conditional`, `production-authorized`, or `rejected`; while permissions are unresolved, keep the item `research-index-only` and express the unknown scopes in the specific permission fields;
- copyright owner or licensor;
- likeness/portrait consent status;
- AI/ML training permission;
- reference conditioning permission;
- derivative/synthetic-output permission;
- commercial advertising permission;
- territory and term;
- exclusivity and endorsement restrictions;
- attribution and share-alike obligations;
- model/checkpoint license dependencies;
- contract or evidence reference;
- expiry and review date.

The production CSV template contains separate fields for reference conditioning, biometric processing, territory, term, exclusivity, endorsement, attribution, share-alike, model dependencies, expiry, and review date. Do not collapse these into a generic `commercial=true` flag.

Use `cleared` or `not-applicable` for `exclusivity_status`, `endorsement_status`, and `model_dependency_status`. Values such as `denied`, `prohibited`, `research-only`, or merely non-empty prose must fail the production gate.

Editorial, rights-managed, Creative Commons, brand-owned, and public-domain labels have different obligations. Inspect the exact asset license and intended use.

## Rejection rules

Reject a source from canonical geometry when it has:

- wrong or unverified identity;
- face smaller than 96 px;
- severe pitch, roll, profile, fisheye, wide-angle proximity, or perspective distortion;
- open-mouth emotion, crying, exertion, or speech when measuring neutral geometry;
- occlusion by medals, hands, microphones, text, masks, or hair;
- strong retouching, facial reshaping, or beauty filters;
- uncertain age state;
- mirrored asymmetry when side-specific features matter.

It may still serve an expression, pose, or gap-analysis role if manually labeled.

## Production gate

Before conditioning, fine-tuning, generation, or publication, confirm:

- subject/likeness authorization for the intended context;
- AI/ML, derivative-output, and commercial-use coverage;
- asset copyright and photographer/agency permission;
- no implied endorsement outside scope;
- no unauthorized family member, minor, or bystander is included;
- base model, face encoder, identity adapter, LoRA, and video model licenses are compatible;
- outputs will be reviewed for misleading context and identity swap.

If any element is missing, stop at research, report the exact gap, and propose licensing or a standardized authorized reshoot.
