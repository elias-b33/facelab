import { describe, expect, it } from "vitest";
import { analyze } from "../src/index";
import { automaticLandmarks } from "../src/harmony/automatic";
import { HARMONY_CATALOG } from "../src/harmony/catalog";
import { seedDocument } from "../src/harmony/landmarks";
import type { HarmonyView, MeasurementDocument } from "../src/harmony/types";
import { extractRatingFeatures, RATING_FEATURE_KEYS, RATING_FEATURE_SCHEMA, RATING_LEGACY_METRICS } from "../src/rating/features";
import { canonicalInput } from "./helpers";

function irisInput() {
  const input = canonicalInput();
  // The canonical OBJ contains 468 vertices. Add a known iris observation at
  // each eye midpoint so tests can independently exercise 478/468 eligibility.
  for (const [outer, inner] of [[33, 133], [263, 362]] as const) {
    const a = input.landmarks[outer]!, b = input.landmarks[inner]!;
    for (let i = 0; i < 5; i++) input.landmarks.push({ x: (a.x + b.x) / 2, y: (a.y + b.y) / 2, z: (a.z + b.z) / 2 });
  }
  return input;
}
const front = () => analyze(irisInput());
function placed(view: HarmonyView): MeasurementDocument {
  const doc = seedDocument(view, 1);
  for (const point of Object.values(doc.points)) { point.source = "manual"; point.reviewed = true; }
  return doc;
}

describe("versioned rating feature extraction", () => {
  it("returns fixed keys and explicit nulls even when no observation exists", () => {
    const features = extractRatingFeatures({});
    expect(features.schemaVersion).toBe(RATING_FEATURE_SCHEMA);
    expect(Object.keys(features.values)).toEqual(RATING_FEATURE_KEYS);
    expect(Object.keys(features.provenance)).toEqual(RATING_FEATURE_KEYS);
    expect(RATING_FEATURE_KEYS).toHaveLength(91);
    expect(new Set(RATING_FEATURE_KEYS).size).toBe(91);
    expect(RATING_FEATURE_KEYS.filter(key => key.startsWith("harmony."))).toEqual(HARMONY_CATALOG.map(def => `harmony.${def.key}`));
    expect(Object.values(features.values).every(value => value === null)).toBe(true);
    expect(features.quality.frontUsable).toBe(false);
    expect(features.quality.blockingReasons).toContain("A front scan is required.");
  });

  it("exports measured geometry independently of subjective bands, sex and aggregate scores", () => {
    const scan = front(), before = JSON.stringify(scan);
    const features = extractRatingFeatures({ front: scan });
    expect(features.quality.frontUsable).toBe(true);
    for (const key of RATING_LEGACY_METRICS) {
      expect(features.values[`legacy.${key}`]).toBe(scan.metrics.find(metric => metric.key === key)!.value);
    }
    const alternative = analyze({ ...irisInput(), sex: "feminine", bandProfile: "calibrated" });
    alternative.overall = 1; alternative.standardized = 1; alternative.overallPercentile = 1;
    for (const metric of alternative.metrics) { metric.score = 1; metric.percentile = 1; }
    expect(extractRatingFeatures({ front: alternative }).values).toEqual(features.values);
    for (const score of ["jawlineDefinition", "eyeSymmetry", "overallSymmetry", "jawSymmetry", "facialThirds", "facialFifths"]) {
      expect(features.values).not.toHaveProperty(`legacy.${score}`);
    }
    expect(JSON.stringify(scan)).toBe(before);
    expect(Object.values(features.values).filter(value => value !== null)).toHaveLength(17);
    expect(features.quality.warnings.join(" ")).toContain("sharpness or exposure");
  });

  it("keeps fifth proportions but excludes extrapolated hairline thirds", () => {
    const scan = front(), features = extractRatingFeatures({ front: scan });
    const fifths = scan.metrics.find(metric => metric.key === "facialFifths")!.detail!.fifthsPct as number[];
    for (let i = 0; i < 5; i++) expect(features.values[`legacy.facialFifths.${i}`]).toBe(fifths[i]);
    for (let i = 0; i < 3; i++) {
      expect(features.values[`legacy.facialThirds.${i}`]).toBeNull();
      expect(features.provenance[`legacy.facialThirds.${i}`]!.reason).toContain("estimate");
    }
  });

  it("leaves nonfinite, low-confidence and duplicate observations missing", () => {
    const scan = front();
    scan.metrics.find(metric => metric.key === "canthalTilt")!.value = NaN;
    scan.metrics.find(metric => metric.key === "lipRatio")!.confidence = .49;
    scan.metrics.push({ ...scan.metrics.find(metric => metric.key === "fwhr")! });
    const features = extractRatingFeatures({ front: scan });
    expect(features.quality.frontUsable).toBe(true);
    for (const key of ["canthalTilt", "lipRatio", "fwhr"]) expect(features.values[`legacy.${key}`]).toBeNull();
    expect(features.values["legacy.jawToCheekbone"]).not.toBeNull();
    expect(JSON.stringify(features)).not.toContain("NaN");
  });

  it("rejects malformed proportion vectors instead of silently filling slots", () => {
    const sparse = new Array<number>(5); sparse[0] = 100;
    for (const vector of [[10, 20, 30, 40], [20, 20, NaN, 20, 20], [-10, 20, 30, 30, 30], [20, 20, 20, 20, 21], sparse]) {
      const scan = front();
      scan.metrics.find(metric => metric.key === "facialFifths")!.detail = { fifthsPct: vector };
      const features = extractRatingFeatures({ front: scan });
      for (let i = 0; i < 5; i++) expect(features.values[`legacy.facialFifths.${i}`]).toBeNull();
    }
  });

  it("does not interchange iris observations and eye-corner proxies", () => {
    const scan = front();
    scan.frame!.pts = scan.frame!.pts.slice(0, 468); scan.frame!.hasIris = false;
    const features = extractRatingFeatures({ front: scan });
    for (const key of ["eyeSeparationRatio", "midfaceRatio", "eyeToMouthAngle", "browPosition"]) expect(features.values[`legacy.${key}`]).toBeNull();
    expect(features.values["legacy.canthalTilt"]).not.toBeNull();
    expect(features.quality.warnings.join(" ")).toContain("Iris landmarks");
  });

  it.each(["scan", "gates", "contradictory-blocking"])("honors %s quality refusal before reading front features", mode => {
    const scan = front();
    if (mode === "scan") scan.ok = false;
    if (mode === "gates") scan.gates.pass = false;
    if (mode === "contradictory-blocking") scan.gates.blocking.push({ code: "blur", severity: "block", message: "Too blurry", retake: "Retake" });
    const features = extractRatingFeatures({ front: scan, harmony: { version: 1, front: placed("front") } });
    expect(features.quality.frontUsable).toBe(false);
    expect(features.quality.blockingReasons.length).toBeGreaterThan(0);
    expect(Object.values(features.values).every(value => value === null)).toBe(true);
  });

  it.each(["coordinates", "dimensions", "scale", "pose", "topology"])("blocks malformed %s even when legacy gates claim success", mode => {
    const scan = front();
    if (mode === "coordinates") scan.frame!.pts[4]!.z = NaN;
    if (mode === "dimensions") scan.frame!.imageHeight = Infinity;
    if (mode === "scale") scan.frame!.scale = 0;
    if (mode === "pose") scan.frame!.yawDeg = NaN;
    if (mode === "topology") scan.frame!.pts = scan.frame!.pts.slice(0, 467);
    const features = extractRatingFeatures({ front: scan });
    expect(features.quality.frontUsable).toBe(false);
    expect(Object.values(features.values).every(value => value === null)).toBe(true);
  });

  it("refuses a degraded scan with no remaining reliable features", () => {
    const scan = front();
    for (const metric of scan.metrics) metric.confidence = .48;
    const features = extractRatingFeatures({ front: scan });
    expect(features.quality.frontUsable).toBe(false);
    expect(features.quality.blockingReasons).toContain("No front features have adequate measurement evidence.");
  });
});

describe("harmony provenance and invariant units", () => {
  it("accepts adequately supported mesh geometry while leaving anatomical estimates missing", () => {
    const document = automaticLandmarks({ view: "front", aspect: .8, imageHeight: 1280, mesh: irisInput().landmarks });
    const features = extractRatingFeatures({ front: front(), harmony: { version: 1, front: document } });
    expect(features.values["harmony.front.bigonial-width"]).not.toBeNull();
    expect(features.provenance["harmony.front.bigonial-width"]!.source).toBe("mesh");
    expect(features.values["harmony.front.nose-bridge-to-nose-width"]).toBeNull();
    expect(features.values["harmony.front.top-third"]).toBeNull();
  });

  it("requires a legitimate leveling reference even for otherwise complete geometry", () => {
    const document = placed("side");
    document.points.porion = { ...document.points.porion!, source: "estimated", reviewed: false, confidence: .99 };
    const features = extractRatingFeatures({ front: front(), harmony: { version: 1, side: document } });
    expect(Object.entries(features.values).filter(([key]) => key.startsWith("harmony.side.")).every(([, value]) => value === null)).toBe(true);
    expect(features.provenance["harmony.side.nasofrontal-angle"]!.reason).toContain("leveling");
    document.points.porion.reviewed = true;
    const reviewed = extractRatingFeatures({ front: front(), harmony: { version: 1, side: document } });
    expect(reviewed.values["harmony.side.nasofrontal-angle"]).not.toBeNull();
    expect(reviewed.provenance["harmony.side.nasofrontal-angle"]!.source).toContain("reviewed-estimated");
  });

  it("does not substitute outer canthi when a pupil reference is absent", () => {
    const document = placed("front");
    delete document.points.pupilL;
    const features = extractRatingFeatures({ front: front(), harmony: { version: 1, front: document } });
    expect(features.values["harmony.front.mouth-width-to-nose-width-ratio"]).toBeNull();
    expect(features.provenance["harmony.front.mouth-width-to-nose-width-ratio"]!.reason).toContain("pupilL");
  });

  it("propagates dependency exclusions without penalizing unrelated features", () => {
    const document = placed("front");
    document.points.alarL = { ...document.points.alarL!, source: "contour", reviewed: false, confidence: .49 };
    const features = extractRatingFeatures({ front: front(), harmony: { version: 1, front: document } });
    expect(features.values["harmony.front.mouth-width-to-nose-width-ratio"]).toBeNull();
    expect(features.values["harmony.front.bigonial-width"]).not.toBeNull();
    document.points.alarL.confidence = .5;
    const better = extractRatingFeatures({ front: front(), harmony: { version: 1, front: document } });
    expect(better.values["harmony.front.mouth-width-to-nose-width-ratio"]).not.toBeNull();
    expect(better.provenance["harmony.front.mouth-width-to-nose-width-ratio"]!.confidence).toBe(.5);
  });

  it("always normalizes distances to face height even with a millimeter calibration", () => {
    const document = placed("front"), scan = front();
    const uncalibrated = extractRatingFeatures({ front: scan, harmony: { version: 1, front: document } });
    document.calibration = { from: "hairline", to: "menton", millimeters: 180 };
    const before = JSON.stringify(document);
    const calibrated = extractRatingFeatures({ front: scan, harmony: { version: 1, front: document, descriptiveOnly: false, ranges: { "front.cupid-s-bow-depth": { lo: 900, hi: 999 } } } });
    expect(calibrated.values["harmony.front.cupid-s-bow-depth"]).not.toBeNull();
    expect(calibrated.values).toEqual(uncalibrated.values);
    expect(JSON.stringify(document)).toBe(before);
  });

  it("keeps scale, aspect and profile mirror transformations invariant", () => {
    const document = placed("side"), scan = front();
    const base = extractRatingFeatures({ front: scan, harmony: { version: 1, side: document } });
    const transformed = { ...document, aspect: 1.3, imageHeight: 2000,
      points: Object.fromEntries(Object.entries(document.points).map(([key, point]) => [key, { ...point, x: 1 - point.x / 1.3 }])) };
    const other = extractRatingFeatures({ front: scan, harmony: { version: 1, side: transformed } });
    for (const [key, value] of Object.entries(base.values)) {
      if (value === null) expect(other.values[key]).toBeNull();
      else expect(other.values[key], key).toBeCloseTo(value, 8);
    }
  });

  it("leaves degenerate and nonfinite optional measurements missing", () => {
    const document = placed("side");
    document.points.orbitale = { ...document.points.porion! };
    const features = extractRatingFeatures({ front: front(), harmony: { version: 1, side: document } });
    expect(features.quality.frontUsable).toBe(true);
    expect(Object.entries(features.values).filter(([key]) => key.startsWith("harmony.side.")).every(([, value]) => value === null)).toBe(true);
    document.points.porion!.x = NaN;
    expect(extractRatingFeatures({ front: front(), harmony: { version: 1, side: document } }).values["harmony.side.nasofrontal-angle"]).toBeNull();
  });
});
