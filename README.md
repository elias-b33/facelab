# Local FaceIQ

Your personal, local fork of [FreeHarmony](https://github.com/Blueturboguy07/freeharmony).
No account, subscription, paid API, photo upload or hosted deployment is needed.

## Deploy your hosted version

The source is configured for Vercel. Import **elias-b33/facelab**, choose
**apps/web** as the Root Directory and use the Next.js preset. Build/install
commands are in apps/web/vercel.json; no API keys or database are required.
See [DEPLOYMENT.md](DEPLOYMENT.md) for the exact settings and verification steps.
Photos and measurement still run in each visitor's browser. Browser histories
are separate for localhost, Vercel previews and your production domain.

## Open the app

Double-click **Start Local FaceIQ.cmd** in this folder. Keep its console window
open, then visit **http://127.0.0.1:3000** in your browser. Press Ctrl+C in the
console to stop it. The server binds to the loopback address, so it is not
exposed to other devices on your network. Dependencies, the face model and the
production build have been prepared in this installation.

Use the same browser and the exact same address to keep your history together.
`localhost:3000`, `127.0.0.1:3000` and another browser have separate browser storage.

## What is included

- The Private Atlas: an immersive portrait campaign, centered wordmark, fullscreen
  editorial navigation, interactive front/profile study and layered personal dossiers.
  The warm ivory, charcoal and rust palette continues throughout the working studio.
  Original fictional campaign portraits, fonts and illustrations are self-hosted;
  image prompts and provenance are in `apps/web/public/images/credits.json`.
- Restrained motion: slow portrait movement with a pause control, image crossfades,
  scroll reveals, contour drawing, measurement counts and tactile dossier transitions.
  Keyboard navigation and the system's reduced-motion preference are supported.
- A consistent working studio: an indexed measurement library with expandable
  formulas, region and perspective filters; a photograph-led collection; numbered
  workspace settings; chapter-based photo/privacy guides; and quieter capture,
  landmark and report controls. The campaign landing page keeps its own layout.
- **71 named measurements: 36 front + 35 side**, covering all current public
  FaceIQ measurement guides, including five supporting definitions.
- Automatic landmark drafts for both front and profile photos. The existing
  478-point MediaPipe mesh works alongside a local hair/skin segmenter and
  image-edge refinement. No new package, paid API or Python service is needed.
  Failed photo quality checks remain visible, and only suitable front photos
  receive the original FreeHarmony harmony index.
- Detailed front and profile reports with formulas, anatomical point lists,
  construction overlays, placement guidance and explicit missing-data states.
- A guided editor for 53 front and 27 side landmarks, zoom, keyboard movement,
  point confirmation, undo and saved review state. A priority review focuses on
  inferred and missing positions. Existing manual/reviewed points survive a
  detection rerun. Unplaced and low-evidence points are never measured.
- Correct image aspect handling, pupil/eye-corner leveling for front views and
  Frankfort leveling plus mirror handling for side views. The selected metric
  shows a deterministic ±2-pixel placement sensitivity envelope.
- Editable working ranges and a descriptive-only mode. Distances use % of face
  height until you supply an actual measured scale; no face size is assumed.
- A searchable measurement library, region/status filters, written observations,
  all-measurement comparisons, CSV and JSON exports, and a self-contained offline
  HTML report. Print / PDF uses the browser's local print dialog.
- Validated version-3 JSON backups include automatic provenance, quality labels,
  named points, review state, scale, notes and custom ranges. Version-1 and
  version-2 backups and existing scans still open.
- Original 18 frontal metrics and five legacy profile measures remain in the
  classic results view, without rewriting the original saved measurements.
- Optional local Ollama integration, off by default. No cloud AI integration.

The original harmony scores inherit FreeHarmony's reference model. They describe
photographed geometry; this fork does not reproduce FaceIQ Labs' proprietary scores or
establish a universal attractiveness scale. Keep lighting, distance and head
position consistent, and review the points before interpreting a change.

The [public definition index](https://www.faceiqlabs.com/support/harmony), checked
on 22 October 2026, contains 71 guides: 33 front and 33 side definitions used in
their Harmony score, plus five supporting definitions (36 front, 35 side in total).
This version implements all 71 as descriptive geometry. The local catalogue,
formulas, working ranges and declared adaptations are in
`packages/engine/src/harmony/`. FaceIQ's private score weights, reference corpus
and AI are not reproduced. Working ranges are editable local conventions, not
validated beauty thresholds or clinical targets.

The numerical audit and its limits are documented in [MATH_AUDIT.md](MATH_AUDIT.md).
Every definition has an independent numerical fixture. Seeded transformation
tests check all values under translation, scaling, rotation, mirroring of a
profile and changes in image aspect. Missing leveling points, invalid angles
and insufficient score coverage produce explicit missing/invalid results.

## Supervised human-rating model

The app now has a separate, local **mean human-rating prediction** path. Open
**Settings → Human-rating model** (`/settings#rating-model`) to import a model
you have trained and evaluated. No human-rating weights are bundled: without a
usable model, the report shows that a prediction is unavailable. The original
geometric harmony index remains separate.

The offline pipeline fits a small ridge model to 16 existing geometric
summaries. It separates people into training, validation, calibration and test
sets, reports errors against a mean-score baseline, preserves the source label
scale, and records refused photos. It does not establish a universal 1–10 looks
scale. See [the complete model specification](docs/RATING_MODEL.md) for the
math, evidence, feature limitations, JSONL schemas and evaluation requirements.
Start with [the dataset/protocol manifest template](calibration/rating-manifest.example.json),
replacing its placeholders with actual source information; it contains no data
or evidence of accuracy.

A [prior repository experiment](docs/RATING_MODEL.md#prior-repository-experiment)
reported London Set test MAE 0.696 on a 1–7 scale versus 0.673 for the mean
baseline, with 18 accepted test people. It did not beat that baseline. Those
results were not rerun for this version, and the earlier artifact needs new
extraction, training and evaluation before it can meet the current contract.

For a local dataset, keep the photo manifest, labels and pixels in the ignored
`calibration/private/` directory and experiments in `calibration/outputs/`.
The example below uses PowerShell from the repository root after preparing the
actual `photos.jsonl` and dataset `manifest.json`. Install
`calibration/requirements-extraction.txt` and
`calibration/requirements-rating.txt` in your Python environment first, and run
`pnpm setup:assets` if the pinned face model is not present.

```powershell
$ratingData = Join-Path (Get-Location).Path "calibration/private"
$ratingOutput = Join-Path (Get-Location).Path "calibration/outputs"
python calibration/extract_rating_landmarks.py --manifest "$ratingData/photos.jsonl" --out "$ratingData/rating-landmarks.jsonl" --pixels-dir "$ratingData/pixels"
pnpm --filter @freeharmony/engine export:rating --input "$ratingData/rating-landmarks.jsonl" --out "$ratingData/features-and-ratings.jsonl"
python calibration/train_rating.py --input "$ratingData/features-and-ratings.jsonl" --manifest "$ratingData/manifest.json" --out "$ratingOutput/rating-model.json" --report "$ratingOutput/evaluation.json" --splits "$ratingOutput/subject-splits.json"
```

Include actual `rating` objects in the photo manifest, or add
`--labels "$ratingData/labels.jsonl"` to the feature-export command for an exact
ID join. Verified person IDs and front views are required. Actual RGBA pixels
are used for production photo-quality checks; missing pixels leave a refused
row in the audit. Extractor/browser preprocessing is not bit-identical and needs
separate deployed evaluation. Synthetic fixtures test software only and are
rejected for ratings of real photographs.

## Create a complete report

1. Choose **Begin a study** and select a front photograph or use the camera. The
   models suggest landmarks automatically, then show a count of image-based and
   inferred positions. Open the dossier to see the report.
2. Use **Check priority points** to inspect inferred and missing details. Move
   any dot, confirm it, and continue. You can also inspect all points and confirm
   checked usable suggestions together. Left/right refers to the image.
3. Add a side photo from the dossier. Either facing direction works. The app
   attempts a profile face mesh and traces the visible silhouette with the local
   segmentation model and image edges. Ear opening, orbital references, corneal
   front and jaw tangents are anatomical estimates and need closer review.
4. Existing reports have **Auto-detect points** / **Detect again** controls.
   Manual corrections, reviewed positions and a measured scale are kept. Save
   changes to persist the draft; export includes unsaved edits too.
5. Export CSV/JSON or download the detailed offline report. Print / PDF creates
   a local PDF through your browser. Complete photo backups are in Settings.

The draft is useful immediately, with provisional labels. Detection is not a
claim that every point is exact: hidden/cropped features remain unplaced, weak
crease suggestions are excluded until confirmed, and a true profile may need
silhouette-assisted eye and ear estimates when the face mesh cannot lock on.

Millimeter distances require a real known distance in the same image plane.
Without it, `% FH` means a percentage of marked vertical hairline-to-chin height.
A reviewed landmark is a user's placement confirmation, not a clinical
validation. The sensitivity envelope moves one coordinate at a time and is not
a statistical confidence interval or a bound on all photo error.

## Local data and privacy

Photos, measurements and corrections are stored in browser localStorage. The app
does not save scans into the project directory or send them to an external
service. Clearing browser site data removes that browser's history.

Use **Settings & backups → Export backup** before clearing browser data or moving
to another browser. Restore merges new scan IDs and keeps existing scans. The
backup includes the core photos, profile, geometry and corrections; optional AI
annotations and AI caches are excluded. Storage errors preserve existing photos
and ask you to export/delete manually. History currently allows up to 40 scans;
the browser's quota may be reached earlier.

This project is inside your OneDrive folder. Source files and exported backups
saved there may sync under your existing OneDrive settings. Save sensitive
backups in a location you choose. The app itself does not initiate cloud sync.

## Development

Node.js 24 and pnpm 11 were used for this setup. From this folder:

```powershell
pnpm dev            # http://127.0.0.1:3000, with live code changes
pnpm test           # engine, advice and application regression tests
pnpm typecheck
pnpm build          # produce the local production build
pnpm start          # serve the production build on loopback
pnpm security:audit
```

The root launcher disables Next.js telemetry in its child process. There are no
external fonts, analytics, payment SDKs or cloud inference SDKs. The browser's
Content Security Policy blocks external resource and API origins. Optional AI
requests are limited to `localhost:11434` or `127.0.0.1:11434`, reject redirects,
and reject cloud tags and Ollama entries marked as remote models.

Ollama is **not installed or started by this app**. If you already run it, enable
it in Settings, select installed local models and use Check local connection.
Photo requests occur only when you explicitly choose an AI report/check.

## Reinstalling or moving the project

Initial installation requires internet access for packages and the two public
vision models. Regular use after setup requires no internet connection.

```powershell
pnpm install --frozen-lockfile --ignore-scripts
pnpm setup:assets
pnpm security:audit
pnpm build
pnpm start
```

The reviewed asset setup copies MediaPipe's installed WASM files and downloads
the pinned version-1 face and SelfieMulticlass models from Google's official
storage only if missing. Their SHA-256 hashes are checked before use. Starting the app never downloads anything.
The workspace configuration disables install lifecycle scripts and enables
package-store integrity checking; `.npmrc` also retains these settings for older
pnpm versions. Review and scan new dependencies before changing the lockfile.

## Security review and attribution

See [SECURITY_REVIEW.md](SECURITY_REVIEW.md) for the upstream revision, scans,
dependency updates, integrity checks, verification and limits. Machine-specific raw audit results
are kept locally in the ignored `.security/` folder. Microsoft Defender is not bundled with the app.

Based on FreeHarmony by Blueturboguy07, upstream revision
`fa54d317dfaf762d8abbaa70d752a5a1fa92f242`, retrieved on 2026-10-21. Original
license: **AGPL-3.0**, retained in [LICENSE](LICENSE). This is a modified personal
fork, with the original Git history and upstream remote preserved. It is not
affiliated with FaceIQ Labs.


## Automatic vision assets and typography

The contour model is Google MediaPipe SelfieMulticlass 256×256 float32, version 1
(Apache-2.0; 16.4 MB). It predicts background, hair, body skin, face skin, clothing
and accessories, and supplies a draft contour rather than bony anatomy. The
[official model card](https://storage.googleapis.com/mediapipe-assets/Model%20Card%20Multiclass%20Segmentation.pdf)
and [Image Segmenter documentation](https://ai.google.dev/edge/mediapipe/solutions/vision/image_segmenter)
describe its intended use and limitations.

SHA-256: c6748b1253a99067ef71f7e26ca71096cd449baefa8f101900ea23016507e0e0.
The original 3.8 MB Face Landmarker model remains unchanged. Both model hashes
are verified by the local launcher; it downloads nothing during startup.

Instrument Serif Regular and Italic come from Google's official fonts repository,
under the SIL Open Font License. The complete license is retained at
apps/web/public/fonts/OFL.txt. Exact source URLs and hashes are recorded in
.security/editorial-fonts.json. Models and fonts were quarantined and scanned
with Microsoft Defender before they were copied into the application. There
were no package installs or lockfile changes for this upgrade.
