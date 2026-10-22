import type { ScanResult } from "../types";
import { ENGINE_VERSION } from "../version";
import { extractRatingFeatures, RATING_FEATURE_KEYS, RATING_FEATURE_SCHEMA } from "./features";
import type { RatingModel, RatingModelValidation, RatingPrediction } from "./types";

type ObjectValue = Record<string, unknown>;
const object = (value: unknown): value is ObjectValue => typeof value === "object" && value !== null && !Array.isArray(value);
const finite = (value: unknown): value is number => typeof value === "number" && Number.isFinite(value);
const positiveInteger = (value: unknown): value is number => finite(value) && Number.isSafeInteger(value) && value > 0;
const text = (value: unknown): value is string => typeof value === "string" && value.trim().length > 0 && value.length <= 4000;
const between = (value: unknown, min: number, max: number): value is number => finite(value) && value >= min && value <= max;

/** Validate imported JSON as data only. Never evaluates code or fetches URLs. */
export function validateRatingModel(value: unknown): RatingModelValidation {
  const errors: string[] = [];
  const fail = (message: string) => { errors.push(message); };
  if (!object(value)) return { valid: false, model: null, errors: ["The model must be a JSON object."] };
  if (value.schemaVersion !== 1 || value.kind !== "facelab-ridge-rating" || value.featureSchema !== RATING_FEATURE_SCHEMA) fail("Unsupported rating model or feature schema.");
  if (!text(value.modelId) || !text(value.createdAt) || !Number.isFinite(Date.parse(String(value.createdAt)))) fail("The model needs an ID and a valid creation date.");
  const target = value.target;
  if (!object(target) || !text(target.label) || !text(target.population) || !text(target.protocol) ||
      !between(target.min, 0, 10) || !between(target.max, 0, 10) || target.min >= target.max) fail("Declare the human rating scale, panel population, and rating protocol.");
  const provenance = value.provenance;
  if (!object(provenance)) fail("Model provenance is missing.");
  else {
    for (const key of ["datasetId", "datasetName", "sourceUrl", "license", "identityProtocol", "splitSeed", "engineVersion"]) {
      if (!text(provenance[key])) fail(`Missing provenance: ${key}.`);
    }
    if (typeof provenance.dataSha256 !== "string" || !/^[a-f0-9]{64}$/i.test(provenance.dataSha256)) fail("A SHA-256 digest of the training records is required.");
    if (provenance.usage !== "research" && provenance.usage !== "commercial-permitted") fail("Declare the dataset usage restriction.");
    if (typeof provenance.synthetic !== "boolean") fail("The model must declare whether its labels are synthetic.");
    if (typeof provenance.sourceUrl !== "string" || !/^https?:\/\/[^\s/?#]+(?:[/?#][^\s]*)?$/i.test(provenance.sourceUrl)) fail("The dataset source must be an HTTP(S) reference.");
  }
  if (!finite(value.intercept) || Math.abs(value.intercept) > 1e6) fail("Invalid regression intercept.");
  const features = value.features;
  if (!Array.isArray(features) || features.length !== RATING_FEATURE_KEYS.length) fail("The model must contain exactly the 16 versioned geometry features.");
  else for (let index = 0; index < RATING_FEATURE_KEYS.length; index++) {
    const feature: unknown = features[index];
    if (!object(feature) || feature.key !== RATING_FEATURE_KEYS[index]) { fail(`Unexpected feature at position ${index}.`); continue; }
    for (const key of ["mean", "scale", "linear", "quadratic", "min", "max"]) if (!finite(feature[key]) || Math.abs(feature[key] as number) > 1e9) fail(`Invalid ${feature.key}.${key}.`);
    if (!finite(feature.scale) || feature.scale <= 1e-12) fail(`Feature ${feature.key} needs a positive scale.`);
    if (finite(feature.min) && finite(feature.max) && finite(feature.mean) && (feature.min > feature.max || feature.mean < feature.min - 1e-9 || feature.mean > feature.max + 1e-9)) fail(`Invalid training range for ${feature.key}.`);
  }
  const calibration = value.calibration;
  if (!object(calibration) || calibration.method !== "split-conformal-absolute-residual" ||
      !between(calibration.coverage, .5, .99) || !positiveInteger(calibration.subjects) ||
      !between(calibration.absoluteErrorQuantile, 0, 10)) fail("Invalid independent calibration summary.");
  else if (Math.ceil((calibration.subjects + 1) * calibration.coverage) > calibration.subjects) fail("Too few calibration people for the requested finite-sample coverage.");
  const evaluation = value.evaluation;
  if (!object(evaluation)) fail("Held-out test results are missing.");
  else {
    for (const key of ["testSubjects", "testRows", "testTotalRows"]) if (!positiveInteger(evaluation[key])) fail(`Invalid evaluation count: ${key}.`);
    if (!finite(evaluation.testUnusableRows) || !Number.isSafeInteger(evaluation.testUnusableRows) || evaluation.testUnusableRows < 0) fail("Invalid abstention count.");
    for (const key of ["mae", "rmse", "baselineMae", "baselineRmse", "meanIntervalWidth"]) if (!between(evaluation[key], 0, 10)) fail(`Invalid evaluation metric: ${key}.`);
    for (const key of ["testCoverage", "intervalCoverage"]) if (!between(evaluation[key], 0, 1)) fail(`Invalid evaluation proportion: ${key}.`);
    for (const key of ["pearson", "spearman"]) if (evaluation[key] !== null && !between(evaluation[key], -1, 1)) fail(`Invalid correlation: ${key}.`);
    if (finite(evaluation.testRows) && finite(evaluation.testSubjects) && evaluation.testRows !== evaluation.testSubjects) fail("Version 1 requires one independent test photo per person.");
    if (finite(evaluation.testRows) && finite(evaluation.testTotalRows) && finite(evaluation.testUnusableRows) &&
        evaluation.testRows + evaluation.testUnusableRows !== evaluation.testTotalRows) fail("Test acceptance and abstention counts do not agree.");
    if (positiveInteger(evaluation.testTotalRows) && finite(evaluation.testRows) && finite(evaluation.testCoverage) &&
        Math.abs(evaluation.testCoverage - evaluation.testRows / evaluation.testTotalRows) > 1e-6) fail("Test coverage does not match its counts.");
    if (finite(evaluation.mae) && finite(evaluation.rmse) && evaluation.rmse + 1e-9 < evaluation.mae) fail("RMSE cannot be below MAE.");
    if (finite(evaluation.baselineMae) && finite(evaluation.baselineRmse) && evaluation.baselineRmse + 1e-9 < evaluation.baselineMae) fail("The baseline error metrics are inconsistent.");
  }
  const quality = value.quality;
  if (!object(quality) || !between(quality.minimumFeatureConfidence, .5, 1) || !between(quality.maxAbsZ, 1, 6)) fail("Feature quality and extrapolation limits are missing or too permissive.");
  const training = value.training;
  if (!object(training) || !positiveInteger(training.subjects) || !positiveInteger(training.validationSubjects) ||
      !finite(training.regularization) || training.regularization <= 0 || !["linear", "quadratic"].includes(String(training.basis))) fail("Invalid training/validation summary.");
  else if (training.basis === "linear" && Array.isArray(features) && features.some(feature => object(feature) && feature.quadratic !== 0)) fail("A linear model cannot have quadratic coefficients.");
  return errors.length ? { valid: false, model: null, errors } : { valid: true, model: value as unknown as RatingModel, errors: [] };
}

export function parseRatingModel(value: unknown): RatingModel {
  const checked = validateRatingModel(value);
  if (!checked.valid) throw new Error(checked.errors.join(" "));
  return checked.model;
}

/** Engineering minimums, not a certificate of general-population validity. */
export function ratingModelReadiness(model: RatingModel): string[] {
  const issues: string[] = [];
  if (model.provenance.synthetic) issues.push("Synthetic test labels cannot produce a real-person rating.");
  if (model.provenance.engineVersion !== ENGINE_VERSION) issues.push("This model uses a different measurement engine. Re-extract features and retrain it.");
  if (model.training.subjects < 40 || model.training.validationSubjects < 15 || model.calibration.subjects < 20 || model.evaluation.testSubjects < 20) issues.push("The independent training, validation, calibration, and test samples are too small.");
  if (!(model.evaluation.mae < model.evaluation.baselineMae)) issues.push("The model did not beat predicting the training-set mean on its held-out test people.");
  return issues;
}

export function predictRating(result: ScanResult, modelValue?: unknown): RatingPrediction {
  if (modelValue === undefined || modelValue === null) return { status: "unavailable", value: null, reason: "no-model", messages: ["A trained human-rating model has not been installed. Geometry alone does not establish a looks rating."] };
  const checked = validateRatingModel(modelValue);
  if (!checked.valid) return { status: "withheld", value: null, reason: "invalid-model", messages: checked.errors };
  const model = checked.model;
  const readiness = ratingModelReadiness(model);
  if (readiness.length) return { status: "withheld", value: null, reason: model.provenance.synthetic ? "synthetic-model" : model.provenance.engineVersion !== ENGINE_VERSION ? "engine-mismatch" : "unvalidated-model", messages: readiness };
  if (result.engineVersion !== model.provenance.engineVersion) return { status: "withheld", value: null, reason: "engine-mismatch", messages: ["Reanalyze the front photo with the model's measurement engine."] };
  const record = extractRatingFeatures(result);
  if (!result.ok || !result.gates.pass) return { status: "withheld", value: null, reason: "photo-quality", messages: ["The front photo did not pass the measurement checks. Retake it before interpreting a rating."] };
  const missing = model.features.filter(feature => record.features[feature.key] === undefined || (record.confidence[feature.key] ?? 0) < model.quality.minimumFeatureConfidence);
  if (!record.quality_pass || missing.length) return { status: "withheld", value: null, reason: "missing-features", messages: [...record.issues, ...missing.map(feature => `The ${feature.key} measurement is missing or not reliable enough.`)] };
  const outside: string[] = [];
  const contributions = model.features.map(feature => {
    const z = (record.features[feature.key]! - feature.mean) / feature.scale;
    if (!Number.isFinite(z) || Math.abs(z) > model.quality.maxAbsZ) outside.push(feature.key);
    return { key: feature.key, value: feature.linear * z + feature.quadratic * z * z };
  });
  if (outside.length) return { status: "withheld", value: null, reason: "outside-training-range", messages: [`The model would extrapolate too far for: ${outside.join(", ")}. A rating is withheld.`] };
  const clip = (value: number) => Math.max(model.target.min, Math.min(model.target.max, value));
  const value = clip(model.intercept + contributions.reduce((sum, feature) => sum + feature.value, 0));
  const q = model.calibration.absoluteErrorQuantile;
  return { status: "predicted", value, modelId: model.modelId, target: model.target,
    interval: { low: clip(value - q), high: clip(value + q), coverage: model.calibration.coverage },
    evaluation: model.evaluation, contributions,
    messages: ["Predicts the declared panel's mean rating of a frontal photograph on the source scale.",
      "Interval coverage applies marginally to comparable accepted photos under the calibration assumptions; it is not a guarantee for this person.",
      "Feature contributions describe this model's arithmetic, not causal effects or recommended facial changes."] };
}
