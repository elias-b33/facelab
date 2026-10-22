import { describe, expect, it } from "vitest";
import {
  analyze,
  ENGINE_VERSION,
  extractRatingFeatures,
  parseRatingModel,
  predictRating,
  RATING_FEATURE_KEYS,
  validateRatingModel,
  type BandProfile,
  type ImageLike,
  type RatingModel,
  type ScanResult,
} from "../src/index";
import { canonicalInput, canonicalIrisInput, FRAME_W, FRAME_H } from "./helpers";

// Generated texture exercises the real exposure/sharpness gates. This is
// numerical software-test evidence, never evidence about real-photo accuracy.
const syntheticPixels: ImageLike = { width: FRAME_W, height: FRAME_H, data: new Uint8ClampedArray(FRAME_W * FRAME_H * 4) };
for (let y = 0; y < FRAME_H; y++) for (let x = 0; x < FRAME_W; x++) {
  const offset = (y * FRAME_W + x) * 4;
  const shade = 70 + ((x + y) % 7) * 20;
  syntheticPixels.data[offset] = syntheticPixels.data[offset + 1] = syntheticPixels.data[offset + 2] = shade;
  syntheticPixels.data[offset + 3] = 255;
}
const scan = (bandProfile: BandProfile = "faceharmony-parity") => analyze(canonicalIrisInput({ image: syntheticPixels, bandProfile }));

/** In-memory protocol fixture. Its deliberately mocked evaluation metadata
 * tests the importer/math branches; it is NOT a trained or deployable model. */
function protocolFixture(result = scan()): RatingModel {
  return {
    schemaVersion: 1,
    kind: "facelab-ridge-rating",
    featureSchema: "facelab-frontal-geometry-v1",
    modelId: "unit-test-protocol-fixture-not-a-trained-model",
    createdAt: "2026-10-22T00:00:00Z",
    target: { label: "Unit test target", min: 1, max: 5, population: "No real people; mocked metadata", protocol: "Unit test arithmetic only" },
    provenance: {
      datasetId: "unit-test-only", datasetName: "Mocked protocol fixture; no real human ratings",
      sourceUrl: "https://example.invalid/unit-test-only", license: "Not for product use",
      usage: "research", identityProtocol: "Mocked for tests only", dataSha256: "a".repeat(64),
      splitSeed: "unit-test", engineVersion: ENGINE_VERSION,
      // False only to exercise the post-readiness arithmetic branch. A separate
      // test verifies that true ALWAYS prevents real-person predictions.
      synthetic: false,
    },
    features: RATING_FEATURE_KEYS.map(key => {
      const mean = result.metrics.find(metric => metric.key === key)!.value;
      return { key, mean, scale: 1, linear: 0, quadratic: 0, min: mean - 2, max: mean + 2 };
    }),
    intercept: 3,
    calibration: { method: "split-conformal-absolute-residual", coverage: .9, absoluteErrorQuantile: .35, subjects: 100 },
    evaluation: {
      testSubjects: 100, testRows: 100, testTotalRows: 100, testUnusableRows: 0, testCoverage: 1,
      mae: .2, rmse: .25, pearson: .7, spearman: .7, baselineMae: .5, baselineRmse: .6,
      intervalCoverage: .91, meanIntervalWidth: .7,
    },
    quality: { minimumFeatureConfidence: .5, maxAbsZ: 6 },
    training: { subjects: 300, validationSubjects: 100, regularization: .1, basis: "quadratic" },
  };
}

function changeFirstMeasurement(result: ScanResult, fields: Record<string, number>): ScanResult {
  return { ...result, metrics: result.metrics.map(metric => metric.key === RATING_FEATURE_KEYS[0] ? { ...metric, ...fields } : metric) };
}

describe("raw feature contract", () => {
  it("exports measurements independently of heuristic score bands", () => {
    const parity = extractRatingFeatures(scan("faceharmony-parity"));
    const calibrated = extractRatingFeatures(scan("calibrated"));
    expect(parity.quality_pass).toBe(true);
    expect(calibrated.features).toEqual(parity.features);
    expect(Object.keys(parity.features)).toEqual([...RATING_FEATURE_KEYS]);
    expect(parity.features).not.toHaveProperty("facialThirds");
    expect(parity.features).not.toHaveProperty("jawlineDefinition");
  });

  it("does not let duplicate or nonfinite measurements masquerade as complete features", () => {
    const result = scan();
    const first = result.metrics.find(metric => metric.key === RATING_FEATURE_KEYS[0])!;
    expect(extractRatingFeatures({ ...result, metrics: [...result.metrics, first] }).quality_pass).toBe(false);
    expect(extractRatingFeatures(changeFirstMeasurement(result, { value: NaN })).quality_pass).toBe(false);
  });

  it("keeps 468-point geometry descriptive without mixing eye-corner proxies into iris-based rating features", () => {
    const input = canonicalInput({ image: syntheticPixels });
    const result = analyze({ ...input, landmarks: input.landmarks.slice(0, 468) });
    expect(result.ok).toBe(true);
    expect(result.frame?.hasIris).toBe(false);
    const record = extractRatingFeatures(result);
    expect(Object.keys(record.features)).toEqual([...RATING_FEATURE_KEYS]);
    expect(record.quality_pass).toBe(false);
    expect(record.issues.join(" ")).toContain("478-point iris topology");
    expect(predictRating(result, protocolFixture())).toMatchObject({ status: "withheld", value: null });
  });

  it("keeps no-image geometry diagnostic without admitting it to fitting or prediction", () => {
    const result = analyze(canonicalIrisInput());
    expect(result.ok).toBe(true);
    expect(result.gates.warnings.some(gate => gate.code === "image-quality-unchecked")).toBe(true);
    const record = extractRatingFeatures(result);
    expect(record.quality_pass).toBe(false);
    expect(Object.keys(record.features)).toEqual([...RATING_FEATURE_KEYS]);
    expect(record.issues.join(" ")).toContain("Photo sharpness and exposure were not assessed");
    expect(predictRating(result, protocolFixture())).toMatchObject({ status: "withheld", value: null });
  });

  it("does not mistake an omitted image warning for evidence that pixels were assessed", () => {
    const result = analyze(canonicalIrisInput());
    result.gates.warnings = result.gates.warnings.filter(gate => gate.code !== "image-quality-unchecked");
    expect(result.gates.jawEdgeSupport).toBeNull();
    expect(extractRatingFeatures(result).quality_pass).toBe(false);
    expect(predictRating(result, protocolFixture())).toMatchObject({ status: "withheld", value: null });
  });

  it("distinguishes completed image assessment from the strength of its jaw-edge observation", () => {
    const result = scan();
    // A scalar zero still records an assessment. The existing per-feature
    // confidence gates handle weak contours; this adapter adds no jaw threshold.
    result.gates.jawEdgeSupport = 0;
    expect(extractRatingFeatures(result).quality_pass).toBe(true);
  });
});

describe("rating-model artifact validation", () => {
  it("accepts a well-formed data-only protocol fixture", () => {
    expect(validateRatingModel(protocolFixture()).valid).toBe(true);
  });

  it("rejects sparse, incomplete, duplicate, reordered, or invented features", () => {
    const variants = [
      new Array(16),
      protocolFixture().features.slice(1),
      protocolFixture().features.map((feature, index, all) => index === 1 ? all[0] : feature),
      [...protocolFixture().features].reverse(),
      protocolFixture().features.map((feature, index) => index === 0 ? { ...feature, key: "inventedFeature" } : feature),
    ];
    for (const features of variants) {
      expect(validateRatingModel({ ...protocolFixture(), features }).valid).toBe(false);
    }
  });

  it.each([NaN, Infinity, -Infinity, 0, -1])("rejects an invalid feature normalization scale (%s)", (scale) => {
    const model = protocolFixture();
    model.features[0]!.scale = scale;
    expect(validateRatingModel(model).valid).toBe(false);
    expect(() => parseRatingModel(model)).toThrow();
  });

  it("rejects nonfinite weights and impossible held-out acceptance counts", () => {
    const model = protocolFixture();
    model.features[0]!.linear = NaN;
    expect(validateRatingModel(model).valid).toBe(false);
    const wrongCount = protocolFixture();
    wrongCount.evaluation.testCoverage = .5;
    expect(validateRatingModel(wrongCount).valid).toBe(false);
  });

  it("rejects unsupported source protocols and nonexistent conformal order statistics", () => {
    const unsafe = protocolFixture();
    unsafe.provenance.sourceUrl = "javascript:alert(1)";
    expect(validateRatingModel(unsafe).valid).toBe(false);
    const interval = protocolFixture();
    interval.calibration.subjects = 20;
    interval.calibration.coverage = .99;
    expect(validateRatingModel(interval).valid).toBe(false);
  });
});

describe("prediction abstention contracts", () => {
  it("returns unavailable without inventing a rating when no model is installed", () => {
    expect(predictRating(scan())).toMatchObject({ status: "unavailable", reason: "no-model", value: null });
  });

  it("always withholds a model explicitly trained on synthetic test labels", () => {
    const model = protocolFixture();
    model.provenance.synthetic = true;
    expect(predictRating(scan(), model)).toMatchObject({ status: "withheld", reason: "synthetic-model", value: null });
  });

  it("withholds a model that fails to beat the training-mean baseline", () => {
    const model = protocolFixture();
    model.evaluation.mae = model.evaluation.baselineMae;
    model.evaluation.rmse = model.evaluation.baselineRmse;
    expect(predictRating(scan(), model)).toMatchObject({ status: "withheld", reason: "unvalidated-model", value: null });
  });

  it("requires sufficient independent validation and test people", () => {
    const model = protocolFixture();
    model.training.validationSubjects = 1;
    expect(predictRating(scan(), model)).toMatchObject({ status: "withheld", reason: "unvalidated-model" });
  });

  it("does not combine a model with features extracted by another engine version", () => {
    const result = scan(), model = protocolFixture(result);
    expect(predictRating({ ...result, engineVersion: "legacy" }, model)).toMatchObject({ reason: "engine-mismatch", value: null });
    model.provenance.engineVersion = "legacy";
    expect(predictRating(result, model)).toMatchObject({ reason: "engine-mismatch", value: null });
  });

  it("withholds failed photos, missing features, and weak feature confidence", () => {
    const result = scan(), model = protocolFixture(result);
    expect(predictRating({ ...result, ok: false }, model)).toMatchObject({ reason: "photo-quality", value: null });
    expect(predictRating({ ...result, metrics: result.metrics.filter(metric => metric.key !== RATING_FEATURE_KEYS[0]) }, model))
      .toMatchObject({ reason: "missing-features", value: null });
    expect(predictRating(changeFirstMeasurement(result, { confidence: .49 }), model))
      .toMatchObject({ reason: "missing-features", value: null });
  });

  it("withholds out-of-domain input instead of clamping it to an impressive rating", () => {
    const result = scan(), model = protocolFixture(result);
    expect(predictRating(changeFirstMeasurement(result, { value: model.features[0]!.mean + 7 }), model))
      .toMatchObject({ reason: "outside-training-range", value: null });
  });
});

describe("prediction arithmetic on the declared source scale", () => {
  it("applies standardized linear/quadratic terms and calibrated residuals without a 1–10 remap", () => {
    const result = scan(), model = protocolFixture(result);
    model.features[0]!.linear = .3;
    model.features[0]!.quadratic = .2;
    const prediction = predictRating(changeFirstMeasurement(result, { value: model.features[0]!.mean + 2, score: 0 }), model);
    expect(prediction.status).toBe("predicted");
    if (prediction.status !== "predicted") return;
    expect(prediction.value).toBeCloseTo(4.4, 10);
    expect(prediction.target).toMatchObject({ min: 1, max: 5 });
    expect(prediction.interval.low).toBeCloseTo(4.05, 10);
    expect(prediction.interval.high).toBeCloseTo(4.75, 10);
    expect(prediction.contributions.reduce((sum, feature) => sum + feature.value, 0)).toBeCloseTo(1.4, 10);
  });

  it("clips the mean and interval to the actual rating scale", () => {
    const result = scan(), model = protocolFixture(result);
    model.intercept = 5.2;
    const prediction = predictRating(result, model);
    expect(prediction.status).toBe("predicted");
    if (prediction.status !== "predicted") return;
    expect(prediction.value).toBe(5);
    expect(prediction.interval).toMatchObject({ low: 4.65, high: 5 });
  });
});
