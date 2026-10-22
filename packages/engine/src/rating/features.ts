import type { MetricKey, ScanResult } from "../types";
import { ENGINE_VERSION } from "../version";
import type { RatingFeatures } from "./types";

export const RATING_FEATURE_SCHEMA = "facelab-frontal-geometry-v1" as const;
/** Keep in sync with calibration/rating_contract.json. Metric values (including
 * documented composite summaries), not the engine's reference-band scores. */
export const RATING_FEATURE_KEYS = [
  "canthalTilt", "eyeSeparationRatio", "eyeSymmetry", "midLowerThird",
  "facialFifths", "midfaceRatio", "fwhr", "jawToCheekbone",
  "chinToPhiltrum", "lipRatio", "mouthToNoseWidth", "eyeToMouthAngle",
  "overallSymmetry", "jawSymmetry", "jawAngularity", "browPosition",
] as const satisfies readonly MetricKey[];

/** The same adapter is used for browser inference and offline training export. */
export function extractRatingFeatures(result: ScanResult): RatingFeatures {
  const features: RatingFeatures["features"] = {};
  const confidence: RatingFeatures["confidence"] = {};
  const issues: string[] = [];
  if (!result.ok || !result.gates.pass) issues.push("The front photo did not pass its measurement checks.");
  if (result.engineVersion !== ENGINE_VERSION) issues.push("Reanalyze this photo with the current engine before rating it.");
  // A 468-point scan uses eye-corner midpoints in place of iris centers. Keep
  // its descriptive geometry, but do not mix those feature meanings with the
  // 478-point detector used for this model's training and browser inference.
  if (result.frame?.hasIris !== true || !Array.isArray(result.frame.pts) || result.frame.pts.length !== 478) {
    issues.push("Rating features require the 478-point iris topology. Reanalyze the photo with the current face detector.");
  }
  // A geometry-only scan can pass the geometric gates, but it has never had
  // exposure or sharpness assessed. Preserve its descriptive measurements
  // below without admitting it to either supervised fitting or prediction.
  // The null image-evidence marker also covers saved results whose warning
  // was omitted; an absent warning alone is not proof that pixels were checked.
  if (result.gates.warnings.some(gate => gate.code === "image-quality-unchecked") ||
      !Number.isFinite(result.gates.jawEdgeSupport)) {
    issues.push("Photo sharpness and exposure were not assessed. Analyze the original photo before requesting a rating.");
  }
  for (const key of RATING_FEATURE_KEYS) {
    const matches = result.metrics.filter(metric => metric.key === key);
    const metric = matches[0];
    if (matches.length !== 1 || !metric || !Number.isFinite(metric.value) ||
        !Number.isFinite(metric.confidence) || metric.confidence < 0 || metric.confidence > 1) {
      issues.push(`Missing or invalid measurement: ${key}.`);
      continue;
    }
    features[key] = metric.value;
    confidence[key] = metric.confidence;
  }
  return { feature_schema: RATING_FEATURE_SCHEMA, engine_version: result.engineVersion,
    quality_pass: issues.length === 0, features, confidence, issues };
}

/** Alias for tooling that exports the numerical feature record. */
export const exportRatingFeatures = extractRatingFeatures;
