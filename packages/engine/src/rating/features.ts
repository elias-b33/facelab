import { analyzeHarmony } from "../harmony/analyze";
import { HARMONY_CATALOG } from "../harmony/catalog";
import type { HarmonySettings, HarmonyView, MeasurementDocument, NamedPoint } from "../harmony/types";
import type { MetricKey, MetricResult, ScanResult } from "../types";

/** Change this version when feature definitions, units, or eligibility change. */
export const RATING_FEATURE_SCHEMA = "facelab-geometry-v1" as const;
const MIN_AUTOMATIC_CONFIDENCE = 0.5;

/** These are geometry values, never the legacy ideal-band scores/percentiles. */
export const RATING_LEGACY_METRICS = [
  "canthalTilt", "eyeSeparationRatio", "midLowerThird", "midfaceRatio", "fwhr",
  "jawToCheekbone", "chinToPhiltrum", "lipRatio", "mouthToNoseWidth",
  "eyeToMouthAngle", "jawAngularity", "browPosition",
] as const satisfies readonly MetricKey[];

// Deliberately fixed instead of automatically expanding when the UI catalog
// grows. A model manifest and its training extractor must share this schema.
const HARMONY_KEYS = [
  "front.top-third", "front.middle-third", "front.lower-third",
  "front.face-width-to-height-ratio", "front.total-facial-width-to-height-ratio",
  "front.midface-ratio", "front.bitemporal-width", "front.cheekbone-height",
  "front.eye-separation-ratio", "front.eye-aspect-ratio", "front.one-eye-apart-test",
  "front.lateral-canthal-tilt", "front.eyebrow-tilt", "front.eyebrow-low-setedness",
  "front.brow-thickness", "front.brow-length-to-face-width-ratio", "front.nose-tip-position",
  "front.intercanthal-nasal-width-ratio", "front.nose-bridge-to-nose-width",
  "front.ipsilateral-alar-angle", "front.midface-jaw-angle-difference", "front.lip-thickness",
  "front.lower-lip-to-upper-lip-ratio", "front.cupid-s-bow-depth",
  "front.mouth-width-to-nose-width-ratio", "front.interpupillary-mouth-width-ratio",
  "front.mouth-corner-position", "front.chin-to-philtrum-ratio", "front.jaw-frontal-angle",
  "front.jaw-slope", "front.bigonial-width", "front.lower-third-proportion",
  "front.ear-protrusion-ratio", "front.ear-protrusion-angle", "front.neck-width",
  "front.upper-eyelid-exposure", "side.browridge-inclination-angle", "side.nasofrontal-angle",
  "side.upper-forehead-slope", "side.facial-convexity-nasion", "side.facial-convexity-glabella",
  "side.total-facial-convexity", "side.facial-depth-to-height-ratio", "side.orbital-vector",
  "side.interior-midface-projection-angle", "side.z-angle", "side.anterior-facial-depth",
  "side.nasal-projection", "side.nasal-tip-angle", "side.nose-tip-rotation-angle",
  "side.nasolabial-angle", "side.nasofacial-angle", "side.nasomental-angle",
  "side.nasal-w-to-h-ratio", "side.frankfort-tip-angle", "side.lower-lip-s-line-position",
  "side.upper-lip-s-line-position", "side.lower-lip-e-line-position", "side.upper-lip-e-line-position",
  "side.lower-lip-burstone-line", "side.upper-lip-burstone-line", "side.holdaway-h-line",
  "side.mentolabial-angle", "side.gonial-angle", "side.mandibular-plane-angle",
  "side.ramus-to-mandible-ratio", "side.gonion-to-mouth-line",
  "side.recession-relative-to-frankfort-plane", "side.submental-cervical-angle",
  "side.eye-protrusion", "side.brow-bossing-angle",
] as const;

export const RATING_FEATURE_KEYS: readonly string[] = Object.freeze([
  ...RATING_LEGACY_METRICS.map(key => `legacy.${key}`),
  ...[0, 1, 2].map(i => `legacy.facialThirds.${i}`),
  ...[0, 1, 2, 3, 4].map(i => `legacy.facialFifths.${i}`),
  ...HARMONY_KEYS.map(key => `harmony.${key}`),
]);

export interface RatingFeatureProvenance {
  /** Heuristic placement quality; neither probability nor prediction accuracy. */
  confidence: number;
  source: string;
  reason?: string;
}
export interface RatingFeatureSet {
  schemaVersion: typeof RATING_FEATURE_SCHEMA;
  /** Missing observations stay null. Never replace missing geometry with zero. */
  values: Record<string, number | null>;
  quality: { frontUsable: boolean; blockingReasons: string[]; warnings: string[] };
  provenance: Record<string, RatingFeatureProvenance>;
}
export interface RatingFeatureInput { front?: ScanResult; harmony?: HarmonySettings }

const finitePositive = (n: number) => Number.isFinite(n) && n > 0;
const validConfidence = (n: unknown): n is number => typeof n === "number" && Number.isFinite(n) && n >= 0 && n <= 1;
const IRIS_DEPENDENT = new Set<string>(["eyeSeparationRatio", "midfaceRatio", "eyeToMouthAngle", "browPosition"]);

function validateFront(front: ScanResult | undefined): string[] {
  if (!front) return ["A front scan is required."];
  const reasons: string[] = [];
  if (!front.ok || !front.gates?.pass) reasons.push("The front scan did not pass measurement quality gates.");
  for (const gate of front.gates?.blocking ?? []) reasons.push(`${gate.code}: ${gate.message}`);
  const frame = front.frame;
  if (!frame) reasons.push("The front measurement frame is missing.");
  else {
    if (!finitePositive(frame.imageWidth) || !finitePositive(frame.imageHeight) ||
      !finitePositive(frame.scale) || frame.scale > Math.hypot(frame.imageWidth, frame.imageHeight)) {
      reasons.push("The front image dimensions or reference scale are invalid.");
    }
    if (!Number.isFinite(frame.rollDeg) || !Number.isFinite(frame.yawAsym) ||
      (frame.yawDeg !== null && !Number.isFinite(frame.yawDeg)) ||
      (frame.pitchDeg !== null && !Number.isFinite(frame.pitchDeg))) reasons.push("The front pose contains nonfinite geometry.");
    if (!Array.isArray(frame.pts) || ![468, 478].includes(frame.pts.length) ||
      (frame.hasIris && frame.pts.length !== 478)) reasons.push("The front landmark topology is invalid.");
    else for (let i = 0; i < frame.pts.length; i++) {
      const p = frame.pts[i];
      if (!p || !Number.isFinite(p.x) || !Number.isFinite(p.y) || !Number.isFinite(p.z)) {
        reasons.push("The front landmarks contain missing or nonfinite coordinates."); break;
      }
    }
  }
  if (!Array.isArray(front.metrics)) reasons.push("The front metric collection is invalid.");
  return [...new Set(reasons)];
}

function metricProblem(metric: MetricResult | undefined): string | undefined {
  if (!metric) return "This metric was not measured.";
  if (!validConfidence(metric.confidence) || metric.confidence < MIN_AUTOMATIC_CONFIDENCE) return "Automatic placement confidence is below 0.5 or invalid.";
  if (metric.flags?.some(flag => /estimated|template|no-image/.test(flag))) return "This metric relies on an estimate or unavailable image evidence.";
  if (!Number.isFinite(metric.value)) return "This metric contains nonfinite geometry.";
}

function pointProblem(point: NamedPoint | undefined): string | undefined {
  if (!point || point.source === "template") return "The landmark is missing or is a template placeholder.";
  if (!["mesh", "contour", "estimated", "manual"].includes(point.source)) return "The landmark source is unknown.";
  if (!Number.isFinite(point.x) || !Number.isFinite(point.y) || point.x < 0 || point.x > 1 || point.y < 0 || point.y > 1) return "The landmark coordinates are invalid.";
  if (point.source === "estimated" && !point.reviewed) return "An unreviewed anatomical estimate is not an observation.";
  if (point.confidence !== undefined && !validConfidence(point.confidence)) return "The landmark confidence is invalid.";
  // Human confirmation is explicit provenance, not a fabricated detector score.
  if (!point.reviewed && point.source !== "manual" &&
    (!validConfidence(point.confidence) || point.confidence < MIN_AUTOMATIC_CONFIDENCE)) return "The landmark lacks adequate automatic placement confidence.";
}

function addHarmonyFeatures(result: RatingFeatureSet, view: HarmonyView, doc: MeasurementDocument | undefined) {
  if (!doc) return;
  const definitions = HARMONY_CATALOG.filter(def => def.view === view && (HARMONY_KEYS as readonly string[]).includes(def.key));
  const commonProblem = view === "front" && !result.quality.frontUsable
    ? "Front quality gates must pass before front harmony geometry can be used."
    : !finitePositive(doc.aspect) || !finitePositive(doc.imageHeight)
      ? "The measurement document dimensions are invalid." : undefined;
  // Use the same observed baseline for every feature. Do not silently fall back
  // from pupils to eye corners, or level a profile using unreviewed ear priors.
  const levelKeys = view === "front" ? ["pupilL", "pupilR"] : ["porion", "orbitale"];
  const levelProblem = levelKeys.map(key => {
    const reason = pointProblem(doc.points[key]); return reason ? `${key}: ${reason}` : undefined;
  }).find(Boolean);
  const documentProblem = commonProblem ?? (levelProblem ? `The leveling reference is unavailable: ${levelProblem}` : undefined);
  // Units are part of the model schema. A user calibration must never turn a
  // training feature expressed in % face height into one expressed in mm.
  const normalized: MeasurementDocument = { ...doc, calibration: undefined };
  const report = documentProblem ? undefined : analyzeHarmony(view, normalized, { descriptiveOnly: true });
  for (const def of definitions) {
    const key = `harmony.${def.key}`;
    const required = [...new Set([...def.required, ...levelKeys])];
    const points = required.map(name => doc.points[name]);
    const sources = [...new Set(points.filter((p): p is NamedPoint => !!p).map(p => p.reviewed ? `reviewed-${p.source}` : p.source))].sort();
    const confidence = points.length ? Math.min(...points.map(p => p && !pointProblem(p) ? p.confidence ?? 1 : 0)) : 0;
    const dependencyProblem = required.map(name => {
      const reason = pointProblem(doc.points[name]); return reason ? `${name}: ${reason}` : undefined;
    }).find(Boolean);
    const measurement = report?.measurements.find(row => row.key === def.key);
    const reason = documentProblem ?? dependencyProblem ??
      (!measurement || measurement.value === null || !Number.isFinite(measurement.value)
        ? measurement?.reason ?? "The measurement construction is missing or invalid." : undefined);
    result.provenance[key] = { confidence, source: sources.join("+") || "missing", ...(reason ? { reason } : {}) };
    if (!reason && measurement) result.values[key] = measurement.value;
  }
  if (view === "side") result.quality.warnings.push("Profile features use reviewed/observed photographic references; they do not establish bone anatomy or verify a true 90-degree camera pose.");
}

/**
 * A deterministic training/inference boundary: raw geometry, null masks and
 * provenance. Passing gates is permission to use a photograph's measurements,
 * not evidence that a looks predictor is accurate. Hand-selected ideals,
 * standardized legacy scores, skin color, inferred sex and estimated hairlines
 * are intentionally absent. Train missing-value handling on training data only.
 */
export function extractRatingFeatures(input: RatingFeatureInput): RatingFeatureSet {
  const result: RatingFeatureSet = {
    schemaVersion: RATING_FEATURE_SCHEMA,
    values: Object.fromEntries(RATING_FEATURE_KEYS.map(key => [key, null])),
    provenance: Object.fromEntries(RATING_FEATURE_KEYS.map(key => [key, { confidence: 0, source: "missing", reason: "No observation was supplied." }])),
    quality: { frontUsable: false, blockingReasons: validateFront(input.front), warnings: [] },
  };
  const front = input.front;
  result.quality.frontUsable = result.quality.blockingReasons.length === 0;
  if (front && result.quality.frontUsable) {
    result.quality.warnings.push(...front.gates.warnings.map(gate => `${gate.code}: ${gate.message}`));
    if (front.gates.jawEdgeSupport === null) result.quality.warnings.push("Image-dependent quality checks were unavailable; landmark geometry alone does not verify sharpness or exposure.");
    if (!front.frame!.hasIris) result.quality.warnings.push("Iris landmarks are unavailable; pupil-centered features are excluded instead of using eye-corner proxies.");
    const find = (key: MetricKey) => {
      const matches = front.metrics.filter(metric => metric.key === key);
      return matches.length === 1 ? matches[0] : undefined;
    };
    for (const name of RATING_LEGACY_METRICS) {
      const metric = find(name), key = `legacy.${name}`;
      const reason = metricProblem(metric) ?? (IRIS_DEPENDENT.has(name) && !front.frame!.hasIris ? "Actual iris landmarks are required for this pupil-centered feature." : undefined);
      result.provenance[key] = { confidence: validConfidence(metric?.confidence) ? metric.confidence : 0, source: "mesh-geometry", ...(reason ? { reason } : {}) };
      if (!reason && metric) result.values[key] = metric.value;
    }
    // Preserve the actual proportions instead of distance from the arbitrary
    // one-third/one-fifth ideals encoded by the legacy index values.
    for (const [name, detail, size] of [["facialThirds", "thirdsPct", 3], ["facialFifths", "fifthsPct", 5]] as const) {
      const metric = find(name), vector = metric?.detail?.[detail];
      const validVector = Array.isArray(vector) && vector.length === size && Array.from(vector).every(value => Number.isFinite(value) && value >= 0 && value <= 100) && Math.abs(vector.reduce((sum, value) => sum + value, 0) - 100) < 1e-3;
      const reason = metricProblem(metric) ?? (!validVector ? "The measured proportion vector is missing or invalid." : undefined);
      for (let i = 0; i < size; i++) {
        const key = `legacy.${name}.${i}`;
        result.provenance[key] = { confidence: validConfidence(metric?.confidence) ? metric.confidence : 0, source: "mesh-proportions", ...(reason ? { reason } : {}) };
        if (!reason && Array.isArray(vector)) result.values[key] = vector[i]!;
      }
    }
    if (!Object.entries(result.values).some(([key, value]) => key.startsWith("legacy.") && value !== null)) {
      result.quality.frontUsable = false;
      result.quality.blockingReasons.push("No front features have adequate measurement evidence.");
    }
  } else for (const key of RATING_FEATURE_KEYS.filter(key => key.startsWith("legacy."))) {
    result.provenance[key]!.reason = "The front scan is missing or failed quality validation.";
  }
  addHarmonyFeatures(result, "front", input.harmony?.front);
  addHarmonyFeatures(result, "side", input.harmony?.side);
  result.quality.warnings = [...new Set(result.quality.warnings)];
  return result;
}
