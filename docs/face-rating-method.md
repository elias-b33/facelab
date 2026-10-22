# FaceLab appearance prediction method

FaceLab can learn to predict the average appearance rating given by a specified group of people to a photograph. Geometry provides useful inputs, but a list of facial proportions does not itself define that rating. The implemented starting point is an automatic landmark pipeline followed by regularized regression trained on independent human ratings, with a separately defined display scale and measured prediction error. This document distinguishes observed project facts, research evidence, and proposed engineering choices. It makes no claim to objective beauty or to FaceIQ Labs' private formula.

The first automatic London experiment is complete. Its 18 accepted test people produced MAE **0.696 on the original 1–7 scale**, compared with **0.673 for the fit-mean baseline**, and Spearman correlation **0.424**. It did not demonstrate an improvement over the constant baseline. The nominal 90% interval covered 15/18 test labels, with wide uncertainty from the small sample. The model remains an opt-in research artifact. Full protocol, reproduction commands and results are in [calibration/rating/README.md](../calibration/rating/README.md).

The shipped implementation uses a fixed identity split of 51/15/17/19 before quality filtering, fit-only mean imputation and standardization, a separate tuning partition for ridge selection, and 17 accepted calibration identities. Input policy thresholds are predefined engineering choices: feature coverage at least 0.9, maximum absolute standardized feature 4, and standardized RMS at most 2.5. They were not optimized on the test set. Grouped nested cross-validation, an appearance encoder and repeated-capture validation below are future work, not completed experiments. Browser inference enforces feature-schema compatibility; the extraction report records detector hashes and versions, but old stored scans do not yet enforce detector-lineage equality at runtime.

The chat review on 22 October 2026 established the following requirements and gaps. The titles identify the original chats; their outcomes were supplied by the coordinating agent after reading them.

| Chat | Finding relevant to this algorithm |
| --- | --- |
| Review rating accuracy | The displayed scores felt too generous. The old presentation divided a geometry index by 10. Its unlabelled calibration corpus has raw median 87.1 and mean 85.887, so an ordinary corpus result can look like an 8.7/10. Those statistics are not human attractiveness labels. |
| Find face-rating training examples | The requested convention is centered near 5, permits 9.5 and 9.7, and makes 10 effectively unattainable. The desired collection is free and eventually includes matched high-resolution front/profile photographs of 100 men and 100 women. The current London collection contains 102 identities and frontal ratings; approved independent strict-scale training labels were still absent. The earlier community reference board does not establish suitable rights, consistent capture, or independent labels. |
| Research FaceIQ Labs and free setup | The app already maps 71 named measurements, including automatic drafts and a correction editor. A complete measurement catalogue is different from an empirically fitted appearance predictor. |
| Find FaceIQ Labs build details | The initial investigation was interrupted; it supplies no validated scoring formula. |

The Henry Cavill comparison also mixed score meanings: the public FaceIQ page shows 7.8 combined **harmony**, 8.5 frontal harmony and 6.6 side harmony. That does not supply an overall appearance training label. Its public documentation describes additional appearance assessments and a power-mean aggregation, but does not establish the accuracy of this local implementation. [Henry Cavill harmony report](https://www.faceiqlabs.com/celebs-wiki/henry-cavill/harmony), [FaceIQ score documentation](https://www.faceiqlabs.com/support/scores).

Direct code inspection confirms that `packages/engine/src/scoring/curve.ts` gives a flat 100 inside selected working bands, and `aggregate.ts` combines those scores using fixed area weights. `norms.json` contains aggregate measurements from 3,125 unlabelled FFHQ photos. Changing the curve, dividing by 10, or converting these values to corpus percentiles can alter presentation; none learns which faces human raters prefer. The independent workbench already contains useful regression and consent-record scaffolding, but its manually supplied appearance rubric, geometric-mean pillar combination, and separate prediction/label rank transforms are not an automatic validated overall model.

The local catalogue contains 36 frontal and 35 profile measurements, including five supporting definitions. Some share the same landmarks or are algebraically related. Some automatic points are explicitly estimated: hairline, eyelid folds, orbital/ear references and jaw tangents are examples. Therefore 71 available values do not mean 71 independent or equally accurate observations. Keep the measurement map and its provenance, but select prediction features by repeatability and held-out performance.

The prediction target must be explicit. For photo `i`, let `r_ij` be independent rater `j`'s response under a fixed question, capture protocol and rating scale:

```text
y_i = mean_j(r_ij)                 # target mean rating of this photo
s_i = sample_sd_j(r_ij)            # disagreement among its raters
SE_i = s_i / sqrt(m_i)             # mean-rating sampling error if ratings are independent
```

Keep the original rating units in training and evaluation. A London 1–7 rating and a community 1–10 reference are different instruments; do not silently pool them. Prefer many independently collected ratings to a few repeated votes. A practical collection design is 25 or more independent ratings per capture, randomized presentation and hidden previous/model scores; that count is a design starting point, not a proven sufficiency threshold. Use self-reported cohort information only for coverage/error audits where appropriate; do not infer sensitive demographic traits from faces.

There is both shared preference and substantial individual disagreement. Germine and colleagues studied reliable individual differences in face preferences, so a group mean is an operational target rather than a universal preference. Jenkins and colleagues found that variation across photographs of the same person exceeded variation between people in attractiveness in their study. This supports separate photo and repeated-capture evaluation, rather than treating one photograph as a stable personal attribute. [Germine et al., 2015](https://pubmed.ncbi.nlm.nih.gov/26441352/), [Jenkins et al., 2011](https://www.sciencedirect.com/science/article/pii/S0010027711002022).

The automatic pipeline should follow this sequence:

1. Detect a single suitable face and measure capture quality, pose, occlusion and expression. Reject a failed capture before rating. MediaPipe supplies estimated landmarks, expression coefficients and transformation matrices; it does not supply attractiveness or calibrated anatomical certainty. [Official Face Landmarker documentation](https://developers.google.com/edge/mediapipe/solutions/vision/face_landmarker).
2. Work in equal image units: convert normalized coordinates to `(x * width, y * height)`, level the pupil baseline and divide lengths by a fixed observed scale such as pupil distance. Version the exact landmark definitions, pose gates and transformations.
3. Extract stable raw ratios, angles, bilateral differences and, where useful, aligned landmark coordinates. Preserve missingness and point provenance. Do not replace unavailable anatomy with an ideal value. Exclude inferred hairline and other weak features from the initial predictor unless their added value survives automatic-capture validation.
4. Train and run the same feature extractor. A model trained on supplied manual `.tem` landmarks cannot be called a validated browser model merely because the browser produces a different face mesh. Such a manual-landmark experiment is a separate research comparison until tested end to end on automatic detections.
5. Predict the human mean and its calibrated interval, then apply the frozen display map. Keep descriptive measurements available independently of rating eligibility.

The initial model can run on a local CPU and export a small deterministic JSON model for browser inference. Standardize raw features using training identities only; use a training-fitted imputation value and explicit missing indicators for optional features if imputation is needed. The portable linear model supports mean imputation (zero after standardization). Reject missing mandatory features. Fit ridge regression directly to the overall human target:

```text
u_ik = (x_ik - training_mean_k) / training_sd_k
phi(u_i) = u_i                                  # first baseline
beta = argmin_b [sum_i (y_i - b0 - b'phi(u_i))^2 + lambda * ||b||^2]
predicted_mean_i = beta0 + beta'phi(u_i)
```

The intercept is unpenalized. Ridge handles correlated measurements through regularization. The implemented pilot chooses from a fixed modest grid using its separate tuning partition. Future experiments can compare a small quadratic expansion or training-fitted PCA within grouped inner folds; more ratios and interactions need not improve prediction. Compare against a training-mean predictor and separately audit the existing geometry index. [Official ridge objective](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html).

Do not derive the overall target by mechanically averaging four invented pillar scores. If separately labelled pillars are useful, learn them as separate targets and evaluate each. A learned joint model can later use front geometry, profile geometry and a frozen image-appearance encoder. Shape alone misses visible texture, grooming, lighting, expression and other image information. The SCUT release includes both shape and appearance modelling and CNN baselines, which motivates testing an appearance branch; it does not guarantee that one will generalize here. Use a model and training images with suitable rights, and retain the simpler model unless an untouched evaluation demonstrates an improvement. [SCUT-FBP5500 official release](https://github.com/HCIILAB/SCUT-FBP5500-Database-Release).

The research API retains the source rating scale and requires explicit research-preview use. Its separate `strictDisplayRating` helper implements the following monotone display convention; the transform is not automatically evidence of a validated 1–10 model:

```text
z = (predicted_mean - training_label_center) / training_label_sd
sigmoid(t) = 1 / (1 + exp(-t))
rating(z) = 1 + 9 * sigmoid(0.495 * z + log(4/5))
```

This anchors `z = 0` at 5, has slope 1.1 there, and approaches 1 and 10 without reaching them for finite `z`. Approximately, `z = -2, -1, 0, 1, 2, 3` maps to `3.06, 3.95, 5.00, 6.11, 7.15, 8.01`. Freeze the center and spread from human training labels, not from the predictions: scaling by prediction variance or predicted ranks would artificially expand weak estimates. A small pilot cannot support the extreme tail just because the formula can represent 9.7. Cap rounded presentation below 10 (for example, at 9.9 with one decimal).

The coefficient 0.495 and this scale shape are product choices, not discovered laws or population rarity. Five is the reference anchor; a nonlinear map does not guarantee an exact arithmetic mean of 5. Do not attach “top 1%” or “one in 100,000” claims without a suitable representative reference study. Keep error in original label units visible in the model report, because compression of a display scale can make displayed MAE look better without improving predictions.

Separate model development, interval calibration and final testing by identity. Every photo, crop, augmentation and profile of one person belongs to the same split. Choose preprocessing, features and hyperparameters using only development data and grouped inner folds. Fit all data-dependent transforms inside each training fold. Reserve calibration identities for uncertainty and untouched test identities for the final assessment; do not repeatedly tune on the test set. With a small pilot, nested cross-validation estimates research performance, but is not an independent deployment certification. [Official grouped validation and training-only preprocessing guidance](https://scikit-learn.org/stable/modules/cross_validation.html).

A simple 90% split-conformal interval uses calibration residuals after the predictor has been frozen:

```text
f_i = clamp(predicted_mean_i, source_min, source_max)
e_i = abs(y_i - f_i)                       # calibration identities only
k = ceil((n_cal + 1) * 0.90)
q = kth_smallest(e)                       # infinity if k > n_cal
raw_interval = [max(source_min, f - q), min(source_max, f + q)]
display_interval = rating(raw_interval endpoints, using the same fixed scale)
```

Use the exact same predictor, including any clipping to source-label bounds, for calibration and inference. This targets coverage for the defined label under exchangeability of calibration and future cases; it does not ensure coverage for every subgroup, every individual rater, or a shifted camera/population. Calibrate and evaluate the same capture acceptance rules. Report rater disagreement separately from model error and landmark sensitivity. A ±2-pixel sensitivity sweep or landmark confidence score is not a 90% prediction interval. [Angelopoulos and Bates, conformal prediction](https://arxiv.org/abs/2107.07511).

Abstain when mandatory features are absent, capture quality fails, feature versions mismatch, or the input is outside the pilot's predefined support policy. The current check uses fixed limits on features standardized with fit-only means and deviations; it catches gross extrapolation, not every possible distribution shift. Report both the accepted-image fraction and accepted-image error so rejection cannot conceal poor coverage. Version every model, dataset manifest, split, scale parameter, calibration residual set and evaluation report.

Repeated photos need their own experiment. Collect at least two sessions per consenting adult, with changes in camera, distance, light and mild pose within the proposed capture rules. Measure prediction differences, rejected-image rates and changes in human mean ratings. A later person-level estimate may robustly aggregate accepted repeated-photo predictions, but must be trained/calibrated against a matching repeated-photo target. Do not claim that a narrow interval from one image establishes stable personal appearance, or that changing one geometric feature will causally improve someone's rating.

The available datasets support a limited pilot. London v5 contains 102 adult identities, 189-point templates and neutral-front 1–7 ratings from 2,513 raters. Its published CC BY 4.0 licence and stated participant consent for studies/research illustration should be retained in the provenance record; that description alone is not an affirmative clearance for every consumer use. Profile images do not come with equivalent profile-rating labels. SCUT-FBP5500 has 5,500 frontal images, labels from 60 volunteers and explicitly restricts use to non-commercial research. Neither establishes the requested commercially cleared first-party cohort or general-population tail frequency. [London v5](https://figshare.com/articles/dataset/Face_Research_Lab_London_Set/5047666/5), [SCUT official terms](https://github.com/HCIILAB/SCUT-FBP5500-Database-Release).

Release criteria must be declared before inspecting the final test. The following are proposed engineering targets, not achieved accuracy claims:

| Criterion | Proposed decision rule |
| --- | --- |
| Basic predictive value | Beat the training-mean baseline on original-unit MAE, with an identity-bootstrap interval for the improvement. Report RMSE, Pearson correlation, Spearman correlation and a calibration plot as well. |
| Useful ordering | Aim for held-out Spearman correlation at least 0.70; disclose the confidence interval and identity count. A higher threshold can be chosen before collecting the test set. |
| Display error | Aim for MAE no greater than 0.5 and 90th-percentile absolute error no greater than 1.0 on the frozen display scale, alongside original-unit errors. These numbers cannot be imported from a different scale. |
| Uncertainty | Report observed coverage and interval width for the nominal 90% interval, with a binomial interval for coverage. A tiny test cannot establish precise coverage. |
| Capture stability | Aim for a 90th-percentile repeat-capture change no greater than 0.5 display points on accepted standardized captures; report failures and changes in human photo ratings separately. |
| Generalization | Publish results for a separate capture source and adequately sized self-reported audit groups. Avoid claiming subgroup accuracy from a few examples. |
| Data readiness | Use documented permitted adult images, independent labels, identity-disjoint manifests and the deployed automatic extractor. Synthetic fixtures verify software only. |

A 102-person study can expose defects and establish whether automatic geometry contains useful predictive signal. It cannot establish precise tail rankings, universal preference, profile accuracy without profile labels, or population-wide robustness. The practical next model is therefore the directly supervised automatic geometry baseline; the substantive improvement after that is a properly labelled, permitted appearance dataset and an independently evaluated image branch.
