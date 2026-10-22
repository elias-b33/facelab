# Automatic rating research baseline

This folder runs a local, supervised research experiment against **published human ratings**. The feature extractor is the same TypeScript engine used by the browser. The output estimates the original **1–7 mean rating of a frontal photograph**. It does not generate a strict population 1–10 rating or four separate pillar scores.

The available evidence does not establish a production appearance model. The fitted artifact remains `researchOnly: true`, and normal browser inference refuses it unless a research caller explicitly opts in.

## Local source audit

| Source | Local evidence | Identity/view support | Use in this experiment |
| --- | --- | --- | --- |
| Face Research Lab London Set | 102 adult faces, 2,513 source ratings per frontal face, 256,326 human responses; 1–7 scale; CC BY 4.0 source metadata | 102 source person IDs; neutral front plus both profiles; 189-point manual templates for front/right profile | Original frontal human means; automatically extract frontal MediaPipe landmarks, with exact pixel gates |
| SCUT-FBP5500 | 5,500 frontal images, 1–5 ratings, 86-point landmarks; source README limits use to noncommercial research | Image IDs do not verify distinct people or prevent cross-identity leakage; no matched profiles | Excluded |
| FEI | 200 source people, 14 views per person | Known within-dataset person IDs; no attractiveness labels | Excluded from supervised rating |
| Community charts/reference boards | Existing manually arranged source examples | No independently collected target rating protocol | Excluded from training labels |

London attribution: DeBruine, Lisa; Jones, Benedict (2017). [Face Research Lab London Set](https://figshare.com/articles/dataset/Face_Research_Lab_London_Set/5047666), doi:10.6084/m9.figshare.5047666. Published participant consent describes lab/web studies and research illustration. Copyright licensing and suitability of a specific commercial product use are separate questions; this experiment does not claim commercial product clearance.

## Reproduce on Windows

Keep data, landmark JSONL, sidecar pixels, models and subject-level evaluation reports outside the repository. The examples below use the user's local data directory. Use the existing app's pinned `face_landmarker.task`, not a newly substituted detector.

```powershell
python -m venv --system-site-packages "$env:LOCALAPPDATA/FaceIQ/rating-validation/.venv"
& "$env:LOCALAPPDATA/FaceIQ/rating-validation/.venv/Scripts/python.exe" -m pip install -r calibration/rating/requirements.txt

& "$env:LOCALAPPDATA/FaceIQ/rating-validation/.venv/Scripts/python.exe" calibration/rating/extract_london.py `
  --data-root "$env:LOCALAPPDATA/FaceIQ/datasets/london" `
  --model "apps/web/public/mediapipe/face_landmarker.task" `
  --out "$env:LOCALAPPDATA/FaceIQ/rating-validation/london-landmarks.jsonl"

pnpm --filter @freeharmony/engine exec tsx scripts/extract-rating-features.ts `
  "$env:LOCALAPPDATA/FaceIQ/rating-validation/london-landmarks.jsonl" `
  "$env:LOCALAPPDATA/FaceIQ/rating-validation/london-features.jsonl"

python calibration/rating/train.py `
  "$env:LOCALAPPDATA/FaceIQ/rating-validation/london-features.jsonl" `
  --out "$env:LOCALAPPDATA/FaceIQ/rating-validation/london-model"

python -m unittest discover -s calibration/rating -p "test_*.py" -v
```

The extractor verifies the detector against the application's exact pinned SHA-256 before doing any extraction. It stores one raw RGBA sidecar per successfully detected photograph so the TypeScript bridge can run the browser's blur/exposure/geometry quality checks. These are local research data, not application assets. The bridge output contains feature numbers, detector/image hashes, engine version and label provenance without pixel arrays. Landmark records retain a local sidecar path for reproduction. Extraction uses two-face detection and rejects both zero-face and multiple-face photographs. Failed detection records remain in the JSONL with null measurements, preserving every source subject in the split and acceptance denominator. No participant identity recognition is performed.

## Frozen evaluation protocol

Subject IDs are assigned to fit/tuning/calibration/test with seed `20261022` before capture-quality filtering. The split fractions are 50%, 15%, 17.5% and the remainder. All captures belonging to one person stay together; repeated captures cannot increase that person's sample weight or calibration count.

Only fit people determine feature availability, feature means/standard deviations and regression weights. The model uses raw geometry and missingness indicators. Columns with less than 90% observed fit-person weight or negligible fit variance are excluded. Thirds inferred from an estimated hairline are not observed measurements, and the shared extractor leaves them null. Missing retained features are mean-imputed in standardized space with a separate learned missingness weight.

The tuning people choose ridge penalty from `0.01, 0.1, 1, 10, 100, 1000` using original-scale MAE. There is no later refit. Predictions are clipped to the original target endpoints before computing calibration residuals and evaluation errors. The comparator is the fit-person mean on the same accepted test people.

The fixed acceptance rule requires usable capture quality, at least 90% retained-feature coverage, maximum observed absolute standardized value at most 4, and observed standardized RMS at most 2.5. This basic range check is a conservative input guard, not validated broad out-of-distribution detection. These thresholds are fixed before reading test scores.

Calibration uses the finite-sample split-conformal absolute residual order statistic `ceil((n + 1) * coverage)`. For repeated accepted captures, a person's maximum residual counts once. The tool refuses to export an interval if there are insufficient independent accepted calibration people for a finite quantile. Intervals are clipped to source-scale endpoints. The nominal 90% coverage is marginal for accepted exchangeable people from this source; it is not a per-person confidence probability or a prediction of the spread of individual rater opinions.

The report includes subject IDs per split, quality failures, selected columns, tuning scores, test MAE/rank correlation, fit-mean baseline, paired person-bootstrap uncertainty, nominal and empirical interval coverage, abstention reasons and selective coverage. Selective thresholds are diagnostic only: choosing one from these test results would reuse the test set for tuning.

## Observed fixed-split result, 22 October 2026

All 102 frontal photographs produced automatic MediaPipe landmarks. The shared engine's pixel and geometry checks accepted 96. The independent test partition contained 19 people, of whom 18 were accepted by the full prediction policy.

| Measure | Result |
| --- | ---: |
| Test MAE, original 1–7 units | 0.696 |
| Fit-mean baseline MAE, same people | 0.673 |
| Test Spearman rank correlation | 0.424 |
| Accepted fraction of test people | 18/19 (94.7%) |
| Accepted calibration people | 17 |
| Nominal 90% interval radius | ±1.059 |
| Observed test interval coverage | 15/18 (83.3%) |
| Bootstrap 95% interval for MAE improvement over baseline | −0.243 to +0.195 |
| Bootstrap 95% interval for Spearman | −0.087 to +0.810 |

This fixed run did **not beat the mean baseline on MAE**, and the uncertainty is wide. It is evidence that the end-to-end automatic training/evaluation path works, not evidence of a relatively accurate or deployable rating system. The seed, gates and model family were not changed in response to the test score. Profiles, texture, new rater panels, representative sampling and external capture conditions remain unvalidated.

Unit tests use explicitly artificial numeric fixtures to check leakage protection, grouping, abstention and interval mathematics. Their scores are never participant accuracy evidence.
