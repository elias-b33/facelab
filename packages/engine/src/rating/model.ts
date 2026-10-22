import type { RatingFeatureSet } from "./features";

/** Portable coefficients produced by calibration/rating/train.py. No executable model code. */
export interface AppearanceRatingModel {
  format: "facelab-rating-linear-v1";
  id: string;
  featureSchema: "facelab-geometry-v1";
  target: { name: string; min: number; max: number; kind: "mean-human-photo-rating" };
  /** This version supports research artifacts only. Changing this flag cannot promote a model. */
  researchOnly: true;
  intercept: number;
  features: { key: string; mean: number; scale: number; weight: number; missingWeight: number }[];
  policy: { minCoverage: number; maxAbsZ: number; maxRmsZ: number };
  calibration: { method: "split-conformal-absolute"; coverage: number; radius: number; subjects: number };
  evaluation: { testSubjects: number; mae: number; baselineMae: number; spearman: number | null; intervalCoverage: number | null; acceptedFraction: number };
  training: { dataset: string; sourceUrl: string; license: string; fitSubjects: number; tuningSubjects: number; calibrationSubjects: number; testSubjects: number; splitSeed: number };
}

export interface AppearanceRatingPrediction {
  status: "predicted" | "unavailable";
  reason: string;
  score: number | null;
  interval: { lo: number; hi: number; nominalCoverage: number } | null;
  modelId: string | null;
  target: AppearanceRatingModel["target"] | null;
  researchOnly: true;
  featureCoverage: number;
  missing: string[];
  outOfDomain: string[];
  /** Linear model terms in source-label units; associations, not causal effects. */
  contributions: { key: string; value: number; imputed: boolean }[];
}

type Obj = Record<string, unknown>;
const object = (value: unknown): value is Obj => !!value && typeof value === "object" && !Array.isArray(value);
const finite = (value: unknown): value is number => typeof value === "number" && Number.isFinite(value);
const count = (value: unknown, min = 0): value is number => finite(value) && Number.isSafeInteger(value) && value >= min;
const fraction = (value: unknown): value is number => finite(value) && value >= 0 && value <= 1;
const string = (value: unknown): value is string => typeof value === "string" && value.length > 0 && value.length <= 1000;

/** Validate imported JSON before it can influence a displayed result. Throws on incompatible artifacts. */
export function parseAppearanceRatingModel(value: unknown): AppearanceRatingModel {
  const fail = (message: string): never => { throw new Error(`Invalid appearance model: ${message}`); };
  if (!object(value)) return fail("expected a JSON object.");
  if (value.format !== "facelab-rating-linear-v1" || value.featureSchema !== "facelab-geometry-v1") return fail("unsupported format or feature schema.");
  if (value.researchOnly !== true) return fail("this format has not been approved for production ratings.");
  if (!string(value.id) || !finite(value.intercept)) return fail("missing identifier or finite intercept.");
  const target = value.target;
  if (!object(target) || !string(target.name) || target.kind !== "mean-human-photo-rating" || !finite(target.min) || !finite(target.max) || target.min >= target.max) return fail("invalid human-label scale.");
  if (!Array.isArray(value.features) || value.features.length < 1 || value.features.length > 256) return fail("expected 1–256 feature coefficients.");
  const keys = new Set<string>();
  for (const feature of value.features) {
    if (!object(feature) || !string(feature.key) || !/^(legacy|harmony)\.[a-zA-Z0-9_.-]+$/.test(feature.key) || keys.has(feature.key)) return fail("duplicate or invalid feature key.");
    keys.add(feature.key);
    if (!finite(feature.mean) || !finite(feature.scale) || feature.scale <= 0 || !finite(feature.weight) || !finite(feature.missingWeight)) return fail("invalid standardization or coefficient.");
  }
  const policy = value.policy;
  if (!object(policy) || !fraction(policy.minCoverage) || policy.minCoverage === 0 || !finite(policy.maxAbsZ) || policy.maxAbsZ <= 0 || !finite(policy.maxRmsZ) || policy.maxRmsZ <= 0) return fail("invalid input acceptance policy.");
  const cal = value.calibration;
  if (!object(cal) || cal.method !== "split-conformal-absolute" || !finite(cal.coverage) || cal.coverage <= 0 || cal.coverage >= 1 || !finite(cal.radius) || cal.radius < 0 || !count(cal.subjects, 1)) return fail("invalid interval calibration.");
  if (Math.ceil((cal.subjects + 1) * cal.coverage - 1e-12) > cal.subjects) return fail("too few independent calibration subjects for a finite interval at this coverage.");
  const ev = value.evaluation;
  if (!object(ev) || !count(ev.testSubjects, 1) || !finite(ev.mae) || ev.mae < 0 || !finite(ev.baselineMae) || ev.baselineMae < 0 || !(ev.spearman === null || (finite(ev.spearman) && Math.abs(ev.spearman) <= 1)) || !(ev.intervalCoverage === null || fraction(ev.intervalCoverage)) || !fraction(ev.acceptedFraction)) return fail("invalid held-out evaluation.");
  const training = value.training;
  if (!object(training) || !string(training.dataset) || !string(training.sourceUrl) || !string(training.license) || !count(training.fitSubjects, 2) || !count(training.tuningSubjects, 1) || !count(training.calibrationSubjects, cal.subjects) || !count(training.testSubjects, ev.testSubjects) || !count(training.splitSeed)) return fail("missing training provenance or inconsistent split counts.");
  // Copy the plain JSON artifact so subsequent caller mutation cannot change validated fields.
  return JSON.parse(JSON.stringify(value)) as AppearanceRatingModel;
}

const clamp = (value: number, lo: number, hi: number) => Math.max(lo, Math.min(hi, value));

/** Pure browser/server inference. Missing data never earns ideal-proportion credit. */
export function predictAppearanceRating(
  input: RatingFeatureSet,
  candidate?: AppearanceRatingModel | null,
  options: { allowResearchModel?: boolean } = {},
): AppearanceRatingPrediction {
  const base: AppearanceRatingPrediction = {
    status: "unavailable", reason: "No validated appearance model is installed.", score: null, interval: null,
    modelId: null, target: null, researchOnly: true, featureCoverage: 0, missing: [], outOfDomain: [], contributions: [],
  };
  if (!candidate) return base;
  let model: AppearanceRatingModel;
  try { model = parseAppearanceRatingModel(candidate); }
  catch (error) { return { ...base, reason: error instanceof Error ? error.message : "Invalid model." }; }
  base.modelId = model.id;
  base.target = model.target;
  if (!options.allowResearchModel) return { ...base, reason: "This model is a research pilot. Enable the local research preview to inspect it." };
  if (input.schemaVersion !== model.featureSchema) return { ...base, reason: "The model and measurement schema do not match." };
  if (!input.quality.frontUsable || input.quality.blockingReasons.length) return { ...base, reason: "A suitable frontal photo is required. " + input.quality.blockingReasons.join(" ") };
  let sum = model.intercept, squaredZ = 0, observed = 0;
  const missing: string[] = [], outOfDomain: string[] = [];
  const contributions: AppearanceRatingPrediction["contributions"] = [];
  for (const feature of model.features) {
    const value = input.values[feature.key];
    if (!finite(value)) {
      missing.push(feature.key);
      sum += feature.missingWeight;
      contributions.push({ key: feature.key, value: feature.missingWeight, imputed: true });
      continue;
    }
    observed++;
    const z = (value - feature.mean) / feature.scale;
    squaredZ += z * z;
    if (!finite(z) || Math.abs(z) > model.policy.maxAbsZ) outOfDomain.push(feature.key);
    const term = z * feature.weight;
    sum += term;
    contributions.push({ key: feature.key, value: term, imputed: false });
  }
  const featureCoverage = observed / model.features.length;
  const report = { ...base, featureCoverage, missing, outOfDomain };
  if (featureCoverage < model.policy.minCoverage) return { ...report, reason: "Too many model measurements are missing or unreliable. Retake the photo or review its points." };
  if (!observed || Math.sqrt(squaredZ / observed) > model.policy.maxRmsZ) outOfDomain.push("combined-feature-distance");
  if (outOfDomain.length || !finite(sum)) return { ...report, reason: "This measurement pattern is outside the pilot model's supported range. No rating is estimated." };
  const score = clamp(sum, model.target.min, model.target.max);
  return {
    ...report, status: "predicted", score,
    interval: { lo: clamp(score - model.calibration.radius, model.target.min, model.target.max), hi: clamp(score + model.calibration.radius, model.target.min, model.target.max), nominalCoverage: model.calibration.coverage },
    reason: "Research estimate of the source panel's mean rating of this frontal photograph. The interval targets similar accepted photographs, not every person's opinion.",
    contributions: contributions.sort((a, b) => Math.abs(b.value) - Math.abs(a.value)),
  };
}
