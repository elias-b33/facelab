# FaceLab rating model: implementation and evidence

## What this version can support

FaceLab can automatically locate visible facial landmarks, calculate geometric
measurements, and apply an imported regression model to predict a **mean human
panel rating of a frontal photograph**. The predictor runs locally in the
browser. Its training pipeline is offline and exports plain JSON, with no
Python service or paid API in the application.

**This checkout does not contain a newly trained, validated attractiveness
model.** No human-rating dataset was supplied in the current checkout for
retraining this implementation. A prior repository experiment reports results
on real human ratings, as documented below; this session did not rerun it.
The current training pipeline, model contract, inference path, checks, and
evaluation tools are implemented. Learned coefficients and measured accuracy
for this version still require an eligible labelled corpus. Synthetic test
fixtures verify software, and must not be presented as evidence of accuracy
on people.

The existing FreeHarmony score remains a geometric reference index. Its range
preferences, falloff curve, and area weights were chosen by the software, rather
than learned from human attractiveness ratings. FFHQ-derived percentiles
describe that particular reference corpus. They do not transform this geometric
index into a calibrated rating of attractiveness or a representative population
percentile. The new supervised rating path is separate.

## Prior repository experiment

The pre-existing [draft PR #1](https://github.com/elias-b33/facelab/pull/1), at
commit [`4c14bdec14b1be1e9ae83b0b1e5203bf6d393bee`](https://github.com/elias-b33/facelab/commit/4c14bdec14b1be1e9ae83b0b1e5203bf6d393bee),
records an automatic geometry experiment on 102 London Set identities. Its
[versioned experiment report](https://github.com/elias-b33/facelab/blob/4c14bdec14b1be1e9ae83b0b1e5203bf6d393bee/calibration/rating/README.md)
reports the following results on the original **1–7 frontal-photo rating scale**.
These are prior reported results, inspected from repository documentation,
not results rerun or independently verified in this implementation session.

| Measure | Prior reported result |
|---|---:|
| Test MAE | 0.696 |
| Training-mean baseline MAE, same accepted people | 0.673 |
| Test Spearman correlation | 0.424 |
| Accepted test people | 18/19 (94.7%) |
| Accepted calibration people | 17 |
| Observed coverage of nominal 90% interval | 15/18 (83.3%) |
| Bootstrap 95% interval for MAE improvement over baseline | −0.243 to +0.195 |

The reported model **did not beat the constant mean baseline**, and its small
test set leaves substantial uncertainty. It does not establish a relatively
accurate or deployable appearance predictor. Its feature schema, missing-data
handling and input gates differ from this version. Its 17 calibration and 18
accepted test people also fall below this version's pilot minimum of 20 each.
Do not import or relabel the prior coefficients as a current model: use the
current automatic extractor, retrain, calibrate and evaluate under a frozen
protocol. Passing the current pilot gates would still not prove generalization.

### Recorded earlier requirements

The prior [method document](https://github.com/elias-b33/facelab/blob/4c14bdec14b1be1e9ae83b0b1e5203bf6d393bee/docs/face-rating-method.md)
also records summaries of four earlier chats, attributed to its coordinating
agent. This is **second-hand evidence from repository documentation**; those
chats were not independently retrieved in this session. The recorded requests
were to correct overly generous displayed ratings; center a desired scale near
5 with 10 effectively unattainable; eventually collect matched high-resolution
front/profile photographs of 100 men and 100 women; and automatically map the
71-measurement catalogue. The recorded investigation did not provide a
validated private scoring formula. These requirements inform the separation of
geometry from prediction, but do not supply new labels. The current contract
retains native rating units; a custom 1–10 target needs its own labels, or an
explicitly frozen display convention that is identified and evaluated as such.

## The prediction target

Let `y` be the average response of a specified panel, using a specified question
and rating scale, to a specified photograph. The useful question is:

> How closely does this model predict that panel's average judgment on new
> people photographed under comparable conditions?

This target is testable. An objective, universal "looks value" is not the label
provided by these datasets. An image model also sees lighting, expression,
grooming and camera perspective. [3] A landmark model measures some photographed
geometry and omits much of that information. The target remains a panel's
judgment of a particular photograph, with the capture conditions recorded.

Retain the source scale in every row and in the artifact. A source rated from
1–5 remains 1–5; a 1–7 study remains 1–7. Do not combine their raw labels. Even
an affine conversion only changes units; it does not establish common anchors
or a shared rater population. The earlier project target of **5 as average and
10 as near-theoretical** therefore needs its own rating protocol or a separately
validated mapping. This pipeline does not manufacture that mapping. It also
does not stretch the highest observed face to 10, equate a percentile with a
looks rating, or claim a corpus endpoint represents perfection.

## Why a geometric baseline is useful, and why it is limited

The original SCUT-FBP5500 study provides a useful feasibility reference. It
includes 5,500 frontal photographs, 86 landmark locations and 1–5 judgments from
60 volunteers. For its whole-dataset geometric baseline, using 18 features and
Gaussian regression, the reported values are correlation 0.6738 and MAE 0.3914.
Its ResNeXt-50 image model has reported correlation 0.8997 and MAE 0.2291.
The geometry and image experiments use different cross-validation protocols, so this is
not a matched ablation. These are published results for those models and that
dataset, **not FaceLab's measured performance**. Correlation 0.90 does not mean
"90% accurate." [1]

The original release explicitly restricts its database to noncommercial
research and its code to academic research. Do not bundle its images or weights
into a product based on an unrelated permissive-looking mirror. [2]

A later primary study analyzes shape and appearance separately and explains why
geometric morphometrics cannot capture skin appearance, while image predictors
can be affected by pose, background and image normalization. Its reported
3.65–5.19% figures concern **shape variance explained by attractiveness** in a
specific multivariate analysis; they must not be misquoted as "geometry explains
only 4% of attractiveness." [3]

The engineering direction is therefore:

1. Keep interpretable geometry for measurement and explanation.
2. Establish a strong, honest baseline against real human labels.
3. Compare a separately trained image model on the identical person-disjoint
   test set.
4. Add an image/geometry ensemble only if held-out results show improvement.

An arbitrary formula such as "jaw 30%, eyes 30%, symmetry 20%, proportions 20%"
cannot establish accuracy. A published measurement can be a feature without
being a universal beauty target. More features also do not automatically improve
prediction; correlated ratios can double-count the same anatomical distances.

## Automatic measurement contract

The shared machine-readable contract is
`calibration/rating_contract.json`, feature schema
`facelab-frontal-geometry-v1`. It uses these **engine measurement outputs**,
before the final reference-band scoring stage, in this order:

| Feature key | Information retained |
|---|---|
| `canthalTilt` | Eye-corner angle |
| `eyeSeparationRatio` | Relative eye separation |
| `eyeSymmetry` | Engineered index combining measured eye differences |
| `midLowerThird` | Relative middle/lower facial height |
| `facialFifths` | Engineered deviation from equal horizontal fifths |
| `midfaceRatio` | Midface proportion |
| `fwhr` | Facial width/height proportion |
| `jawToCheekbone` | Relative jaw width |
| `chinToPhiltrum` | Chin/philtrum proportion |
| `lipRatio` | Relative lip height |
| `mouthToNoseWidth` | Mouth/nose width proportion |
| `eyeToMouthAngle` | Eye/mouth geometry |
| `overallSymmetry` | Engineered index of photographed bilateral differences |
| `jawSymmetry` | Engineered index of photographed jaw differences |
| `jawAngularity` | Frontal soft-tissue contour angle proxy |
| `browPosition` | Brow/eye position |

These are the existing engine's definitions, not alternate formulas written in
Python. Training must extract them using the production TypeScript engine.
They are **16 engineered summaries, not 16 independent anatomical observations**:
several share landmarks, symmetry quantities use fixed weighting and clipping,
and the fifths feature encodes deviation from equal widths. The model can learn
whether these summaries predict panel ratings, but it cannot recover information
already discarded by their construction. `jawAngularity` is a frontal contour
proxy, not a clinical measurement of the bony gonial angle. Detector depth is
model-estimated relative depth, not a calibrated 3D facial scan. [6] Future models
should compare less compressed coordinates or distances as well as these
summaries, using the same held-out people.
`facialThirds` is excluded because the legacy definition estimates the hairline;
`jawlineDefinition` is excluded because its image-dependent behavior differs
when pixels are absent. A detailed report can still show those measurements
with their existing qualifications.

The model requires the current feature schema and matching engine version.
It also requires the 478-point iris mesh. Legacy 468-point scans can retain
descriptive geometry, but their eye-corner midpoint proxies cannot substitute
for observed iris centers under this rating feature contract.
Do not reuse fitted coefficients after changing measurement units, landmark
definitions, or formulas without re-extracting and evaluating the dataset.
The 71-measurement front/profile catalogue remains useful, but profile
measurements are not automatically inputs to this frontal predictor. A side
photograph does not acquire a human rating merely because the same person's
front photograph has one.

Photo gates must pass. Each required measurement must exist, be finite and have
at least 0.5 engine confidence. This confidence is an engineering estimate of
measurement support, not a calibrated probability that the value is correct.
Missing geometry is never replaced with an ideal measurement or zero. An
otherwise accepted photograph with a required standardized feature magnitude
`abs(z) > 6` is withheld as outside the supported input range. This is
a conservative numerical guard, not a proven detector of every unfamiliar face.
Training min/max values are descriptive and do not form extra hard cutoffs.

## Model mathematics

Training estimates each feature's mean `mu[j]` and standard deviation `s[j]`
using **training people only**:

```text
z[j] = (x[j] - mu[j]) / s[j]
u(x) = intercept + sum(linear[j] * z[j] + quadratic[j] * z[j]^2)
prediction(x) = clip(u(x), target.min, target.max)
```

The linear candidate sets every quadratic coefficient to zero. The additive
quadratic candidate can learn a preferred intermediate range for a proportion
without assuming where that range lies. It has no arbitrary golden-ratio peak.
It is still a small model and cannot represent all interactions among features.
A constant training feature gets a denominator of 1 and an inert fitted term.

For each candidate basis, ridge regression minimizes:

```text
mean((y - intercept - design(z) @ coefficients)^2)
  + lambda * sum(coefficients^2)
```

The intercept is not penalized. The implementation evaluates lambda values
`0.01, 0.1, 1, 10, 100`; validation MAE chooses the model. RMSE breaks a tie,
then smaller basis size and stronger regularization. A training-only transform
prevents information from validation/calibration/test people affecting the
fit. Scikit-learn's official guidance explains why fitting preprocessing before
the split or repeatedly tuning on a test set produces misleading results. [4]

The model stores floating-point coefficients rather than rounding them to
display precision. The browser can round a displayed prediction; calculations
and calibration use the unrounded value.

## Four independent sets

Version 1 deliberately accepts **one preselected frontal photograph per verified
person**. Duplicate photo IDs or duplicate person IDs fail validation. The
curator must document how identities were verified. Filename uniqueness alone
is not evidence that two images depict different people.

Subject IDs are ordered by SHA-256 of `seed + NUL + subject_id`, then assigned
without consulting labels. For `n` people, validation receives at least 15
people or 15%; calibration receives at least 20 or 15%; test receives at least
20 or 20%; training receives the remainder. At least 100 total people and
40 usable training people are required. These thresholds are **pilot engineering
minimums**, not proof of sufficient sample size for deployment.

| Set | Permitted use |
|---|---|
| Training | Estimate feature transforms, fit coefficients and the constant baseline |
| Validation | Choose basis and regularization |
| Calibration | Estimate the final prediction interval width |
| Test | Evaluate the frozen model and interval once |

Photo/feature-quality failures remain in the split audit as abstentions. The
fixed standardized-distance gate is then applied to validation, calibration,
test and runtime. The training fit uses all training rows that pass photo and
feature quality. Each set's usable-person minimum applies after exclusions;
the initial 100-person minimum includes refused records. The report gives each
split's total, accepted and refused rows, refusal reasons and accepted fraction.
It does not silently report excellent performance on a tiny accepted subset.

Keep the seed and person selection fixed before inspecting outcomes. Changing
the seed until the test looks good, feeding test outcomes back into weights, or
refitting on the calibration set invalidates the corresponding evaluation.
Use additional independent people for the next final test if you make changes
based on a previous final test.

## Uncertainty

The selected predictor produces absolute residuals on the untouched calibration
people. For requested coverage `c` and `n` calibration people:

```text
k = ceil((n + 1) * c)
q = kth-smallest absolute calibration residual
interval(x) = [max(target.min, prediction(x) - q),
               min(target.max, prediction(x) + q)]
```

The order statistic is not interpolated. At least 20 usable calibration people
are required as a **pilot engineering floor**, not a universal mathematical
minimum. Independently, the trainer refuses a coverage request for which
`k > n`, because the requested finite residual order statistic does not exist.
Default requested coverage is 90%; the shared format accepts coverage from
50% to 99%.

Split conformal provides marginal coverage under exchangeability of calibration
and future cases. It does not promise 90% correctness for each person or subgroup
and does not survive arbitrary changes in camera protocol, audience or data
distribution. Here it is conditional on the fixed acceptance gates and concerns
the panel mean rating, not every individual rater's opinion. Actual test coverage
and interval width must be reported beside the nominal target. [5]

Do not confuse this with the detailed measurement report's deterministic
±2-pixel sensitivity envelope. That envelope describes changes in placement;
it is not a statistical confidence interval for attractiveness.

## Prepare data and train

Use a corpus you are entitled to use for the intended purpose and retain its
human labels, subject IDs, rating counts and protocol. This script downloads
nothing and does not verify declarations on your behalf.

Create a manifest such as the following **template**. Replace the descriptions
with the actual source; do not label unknown identities or synthetic judgments
as verified human data. A copyable template is
[`calibration/rating-manifest.example.json`](../calibration/rating-manifest.example.json).
Its `synthetic:false` field is a declaration, not proof of human labels. The
placeholder source URL and all `REPLACE...` fields must be replaced.

```json
{
  "datasetId": "your-consented-study-v1",
  "datasetName": "Your actual study name",
  "sourceUrl": "https://example.invalid/replace-with-your-actual-source",
  "license": "Actual permission and licence reference",
  "usage": "research",
  "identityProtocol": "How unique people were verified and one front photograph selected",
  "synthetic": false,
  "featureSchema": "facelab-frontal-geometry-v1",
  "engineVersion": "0.2.0-rating",
  "target": {
    "label": "Mean panel rating",
    "min": 1,
    "max": 5,
    "population": "The actual rater population and intended scope",
    "protocol": "The actual question, anchors, photograph conditions and rater instructions"
  }
}
```

The joined JSONL input has one record per verified person. `features` and
`confidence` must contain the 16 ordered-contract keys. The abbreviated example
below is documentation, not a valid complete training record:

```json
{
  "id": "photo-001",
  "subject_id": "verified-person-001",
  "identity_verified": true,
  "view": "front",
  "feature_schema": "facelab-frontal-geometry-v1",
  "engine_version": "0.2.0-rating",
  "quality_pass": true,
  "features": {"canthalTilt": 4.1},
  "confidence": {"canthalTilt": 0.9},
  "rating": {"mean": 3.2, "count": 30, "scale": [1, 5]}
}
```

At least five independent human ratings per photo are required by this initial
pipeline. That threshold alone does not establish good panel reliability. More
raters, repeated-rating consistency checks, split-half reliability, and direct
estimation of uncertainty in the target mean are preferable.

### 1. Extract local photographs and actual pixels

Create a private **photo manifest JSONL** with `id`, `subject_id`,
`identity_verified:true`, `view:"front"`, `image_path`, and an optional inline
`rating` object. Keep one preselected photograph per independently verified
person. `image_path` may be absolute or relative to the photo manifest's own
directory. This photo manifest is separate from the dataset/protocol JSON above.
It contains the actual source records; the checked-in template supplies none.

Install the offline extractor's requirements and the training dependency in an
environment you control. Use the existing asset setup to obtain the pinned
Face Landmarker task when it is not already present:

```bash
python -m pip install -r calibration/requirements-extraction.txt
python -m pip install -r calibration/requirements-rating.txt
pnpm setup:assets
```

The scripts themselves do not download images, labels, identities or models.
The explicit asset setup command downloads the public application models as
described in the README. Then extract the supplied local photographs:

```bash
python calibration/extract_rating_landmarks.py \
  --manifest /your/private/data/photos.jsonl \
  --out /your/private/data/rating-landmarks.jsonl \
  --pixels-dir /your/private/data/pixels
```

The extractor also accepts `--model /absolute/path/face_landmarker.task` and
`--max-long-edge 2048`. The default maximum long edge is 2048, matching the
browser upload limit. It applies EXIF orientation, uses the pinned task model,
retains full-precision landmark values and writes RGBA bytes with `rgba_path`
references. Detection and image-read failures remain records so subsequent
reports can disclose abstentions. Extraction refuses an existing output file;
choose a new output filename for a new run.

This reproduces the feature and quality-code path, **not bit-identical browser
detection**. Pillow decoding/resizing, browser interpolation, detector backends,
and the application's stored JPEGs can produce different pixels and landmark
outputs. Record that preprocessing distinction and evaluate browser-produced
photos before claiming deployed accuracy.

### 2. Run the production feature and quality code

The feature exporter accepts landmark records with `id`, `subject_id`,
`identity_verified:true`, `view:"front"`, `w`, `h`, `landmarks`, `rgba_path`, and
optional `matrix`/`blendshapes`. Actual RGBA bytes are necessary to run production
blur/exposure and image-support gates. A row without pixels is retained with
`quality_pass:false`; no synthetic image or blanket passing quality report is
substituted. A supplied `rgba_path` that cannot be read or whose byte length
does not match `w * h * 4` is an integrity error and stops the export.

When ratings were included in the photo manifest, run:

```bash
pnpm --filter @freeharmony/engine export:rating \
  --input /your/private/data/rating-landmarks.jsonl \
  --out /your/private/data/features-and-ratings.jsonl
```

For a separate label file, add `--labels /your/private/data/labels.jsonl`.
Each label row has `id` and `rating:{mean,count,scale}`. Matching is by exact
photo ID; missing, duplicate and unused labels are errors. No labels are
inferred from geometry, filenames or another application's proprietary score.
Use **absolute paths** for this command because pnpm runs the filtered script
from the engine package directory. Relative `rgba_path` values are resolved
against the input landmark file's directory.
The exporter also refuses an existing destination file, so choose a new export
filename when rerunning an experiment.

Existing `calibration/extract_landmarks.py` output needs a genuine
source-to-person mapping, matching source labels, a declared front view and
actual image pixels before it can be used here. Image IDs must not be renamed
to supposedly verified person IDs without an audit.

### 3. Train and inspect the held-out report

```bash
python calibration/train_rating.py \
  --input /your/private/data/features-and-ratings.jsonl \
  --manifest /your/private/data/manifest.json \
  --out /your/private/output/rating-model.json \
  --report /your/private/output/evaluation.json \
  --splits /your/private/output/subject-splits.json
```

The output model contains transformations, coefficients, declared source scale,
provenance, test metrics, and interval calibration. It contains no images or
subject IDs. The separate audit report and split assignment include local IDs
and should stay with the private dataset. Model/data hashes support reproducible
comparison; they are not signatures or independent verification of claimed
metrics. An edited artifact can make false declarations, so review the actual
training/evaluation process before distributing a model.

### 4. Import the model in the local app

Open **Settings → Human-rating model** (`/settings#rating-model`) and import
the resulting `rating-model.json`. The app reports model provenance and
declared held-out evidence, then predicts on compatible front reports that pass
the runtime gates. The prediction is separate from the original geometric
harmony index. With no usable model installed, the interface shows the
no-model state rather than generating a looks number from reference-band
weights. No learned coefficients are bundled by default.

For work inside this repository, `calibration/private/` and
`calibration/outputs/` are ignored by Git. Keep photographs, pixel buffers,
human labels, person assignments and model experiments there or in a private
location outside the repository. Ignored paths help prevent accidental commits;
they are not an access-control mechanism. Do not place these records in the
public web assets directory.

## Acceptance and next experiments

The held-out report includes MAE, RMSE, Pearson and Spearman correlations,
median and 90th-percentile absolute error, nominal and observed interval
coverage, interval width, and abstention counts. Errors remain in the **source
rating units**. The constant baseline predicts the training-label mean and is
evaluated on exactly the same accepted test people. A model that cannot improve
on that baseline is not useful enough to publish as a rating predictor.

The importer distinguishes a well-formed artifact from an empirically
promising one. Synthetic models, mismatched schema/engine versions, invalid
numbers, insufficient test/calibration counts and failure to improve on the
mean baseline must not become real ratings. Passing these checks is still not
a certification of broad accuracy. The importer checks the declared evidence;
it cannot independently establish that the supplied report was produced by the
claimed experiment. Its pilot sample and baseline gates are implementation
defaults, not a validated product performance standard. Set any stricter MAE,
coverage, refusal-rate and external-test release requirements in advance for
the actual intended use, and document changes to those requirements.

Before claiming the system is relatively accurate, run these experiments:

- **Performance uncertainty:** report person-level bootstrap confidence
  intervals for improvement over the mean baseline and for rank correlation.
  The current trainer's point estimates and conformal prediction intervals do
  not quantify uncertainty in those performance statistics. The earlier London
  draft illustrates why that additional analysis matters for a small study.
- **Ablations:** mean baseline, geometry-only model, image-only model and learned
  combination, with identical subjects and labels in each split.
- **Independent photographs:** new people and repeated photos of the same
  people under documented capture changes; measure within-person prediction
  variation separately from label-prediction error.
- **External generalization:** a second eligible study with its own raters and
  camera conditions, retaining its original scale or a justified linked scale.
- **Error distribution:** inspect score ranges and image conditions with enough
  cases, not just an average. If self-reported group information is legitimately
  available for evaluation, audit performance without inferring protected
  attributes from faces or using them as hidden score modifiers.
- **Intervals and refusal:** measure both observed interval coverage and the
  fraction of inputs accepted, including difficult photographs.

For an actual product trained beyond this baseline, collect a sufficiently
large, consented corpus representing the intended use, with many independent
raters and a clear question. Compare small pretrained image encoders with
fine-tuning, and learn ensemble weights on validation data. Keep geometry as an
explanation of what was measured; do not present it as a causal explanation of
why a person has a particular attractiveness rating. Neither ridge coefficients
nor image attention maps establish a beneficial cosmetic intervention.

## Sources

1. Liang et al. (2018), *SCUT-FBP5500: A Diverse Benchmark Dataset for
   Multi-Paradigm Facial Beauty Prediction*.
   [Original paper](https://arxiv.org/pdf/1801.06345).
2. HCIILAB, *SCUT-FBP5500 Database Release*, original release and usage terms.
   [Author repository](https://github.com/HCIILAB/SCUT-FBP5500-Database-Release).
3. Sano & Kawabata (2023), *A computational approach to investigating facial
   attractiveness factors using geometric morphometric analysis and deep
   learning*. Scientific Reports 13, 19797.
   [Article](https://www.nature.com/articles/s41598-023-47084-x).
4. Scikit-learn, *Common pitfalls and recommended practices* and
   *Cross-validation: evaluating estimator performance*.
   [Preprocessing and leakage](https://scikit-learn.org/stable/common_pitfalls.html),
   [cross-validation](https://scikit-learn.org/stable/modules/cross_validation.html).
5. Angelopoulos & Bates, *A Gentle Introduction to Conformal Prediction and
   Distribution-Free Uncertainty Quantification*.
   [Authors' paper](https://arxiv.org/abs/2107.07511).
6. Google, *MediaPipe Face Mesh V2 Model Card*, including relative depth,
   synthetic depth training and documented operating limits.
   [Official model card](https://storage.googleapis.com/mediapipe-assets/Model%20Card%20MediaPipe%20Face%20Mesh%20V2.pdf).
