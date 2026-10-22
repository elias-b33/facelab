import { describe, expect, it } from "vitest";
import { parseAppearanceRatingModel, predictAppearanceRating, type AppearanceRatingModel } from "../src/rating/model";
import { strictDisplayRating, formatStrictDisplayRating } from "../src/rating/scale";
import type { RatingFeatureSet } from "../src/rating/features";

const fixture = (): AppearanceRatingModel => ({
  format: "facelab-rating-linear-v1", id: "synthetic-contract-test", featureSchema: "facelab-geometry-v1",
  researchOnly: true, target: { name: "Synthetic panel mean", min: 1, max: 7, kind: "mean-human-photo-rating" }, intercept: 4,
  features: [
    { key: "legacy.canthalTilt", mean: 0, scale: 2, weight: 0.5, missingWeight: -0.2 },
    { key: "legacy.fwhr", mean: 2, scale: 0.5, weight: -0.25, missingWeight: 0.1 },
  ],
  policy: { minCoverage: 0.5, maxAbsZ: 4, maxRmsZ: 3 },
  calibration: { method: "split-conformal-absolute", coverage: 0.9, radius: 0.6, subjects: 18 },
  evaluation: { testSubjects: 20, mae: 0.4, baselineMae: 0.7, spearman: 0.6, intervalCoverage: 0.9, acceptedFraction: 0.8 },
  training: { dataset: "Synthetic test only", sourceUrl: "https://example.org/synthetic", license: "Synthetic", fitSubjects: 50, tuningSubjects: 15, calibrationSubjects: 18, testSubjects: 20, splitSeed: 42 },
});
const input = (values: Record<string, number | null> = { "legacy.canthalTilt": 2, "legacy.fwhr": 2.5 }): RatingFeatureSet => ({
  schemaVersion: "facelab-geometry-v1", values,
  quality: { frontUsable: true, blockingReasons: [], warnings: [] }, provenance: {},
});

describe("portable appearance predictor", () => {
  it("has no fallback rating when a trained model is absent, and research is explicitly opt-in", () => {
    expect(predictAppearanceRating(input()).score).toBeNull();
    expect(predictAppearanceRating(input(), fixture()).status).toBe("unavailable");
  });
  it("predicts in the source label scale using standardized raw features", () => {
    const result = predictAppearanceRating(input(), fixture(), { allowResearchModel: true });
    expect(result.status).toBe("predicted");
    expect(result.score).toBeCloseTo(4.25);
    expect(result.interval?.lo).toBeCloseTo(3.65);
    expect(result.interval?.hi).toBeCloseTo(4.85);
    expect(result.target?.max).toBe(7);
    expect(result.contributions.map(x => x.value)).toEqual([0.5, -0.25]);
  });
  it("uses training mean plus learned missingness term, never an ideal score", () => {
    const result = predictAppearanceRating(input({ "legacy.canthalTilt": null, "legacy.fwhr": 2.5 }), fixture(), { allowResearchModel: true });
    expect(result.score).toBeCloseTo(3.55);
    expect(result.featureCoverage).toBe(0.5);
    expect(result.missing).toEqual(["legacy.canthalTilt"]);
    expect(result.contributions.find(x => x.key === "legacy.canthalTilt")?.imputed).toBe(true);
  });
  it("rejects insufficient data, failed photo gates and stale feature schemas", () => {
    const model = fixture();
    model.policy.minCoverage = 0.9;
    expect(predictAppearanceRating(input({}), model, { allowResearchModel: true }).score).toBeNull();
    const failed = input(); failed.quality.frontUsable = false;
    expect(predictAppearanceRating(failed, model, { allowResearchModel: true }).score).toBeNull();
    const blocked = input(); blocked.quality.blockingReasons.push("Retake");
    expect(predictAppearanceRating(blocked, model, { allowResearchModel: true }).score).toBeNull();
    const incompatible = { ...input(), schemaVersion: "future" } as unknown as RatingFeatureSet;
    expect(predictAppearanceRating(incompatible, model, { allowResearchModel: true }).reason).toContain("schema");
  });
  it("abstains on a single extreme or jointly distant vector before clipping", () => {
    const single = predictAppearanceRating(input({ "legacy.canthalTilt": 20, "legacy.fwhr": 2 }), fixture(), { allowResearchModel: true });
    expect(single.score).toBeNull(); expect(single.outOfDomain).toContain("legacy.canthalTilt");
    const joint = predictAppearanceRating(input({ "legacy.canthalTilt": 7, "legacy.fwhr": 3.75 }), fixture(), { allowResearchModel: true });
    expect(joint.score).toBeNull(); expect(joint.outOfDomain).toContain("combined-feature-distance");
  });
  it("clamps the prediction and interval consistently with the trainer", () => {
    const model = fixture(); model.intercept = 20;
    const result = predictAppearanceRating(input(), model, { allowResearchModel: true });
    expect(result.score).toBe(7); expect(result.interval?.lo).toBe(6.4); expect(result.interval?.hi).toBe(7);
  });
  it("validates untrusted artifacts and prevents removing the research designation", () => {
    for (const mutate of [
      (model: AppearanceRatingModel) => { model.features[0]!.scale = 0; },
      (model: AppearanceRatingModel) => { model.features[0]!.weight = Infinity; },
      (model: AppearanceRatingModel) => { model.features[1]!.key = model.features[0]!.key; },
      (model: AppearanceRatingModel) => { model.calibration.subjects = 4; },
      (model: AppearanceRatingModel) => { model.training.calibrationSubjects = 1; },
      (model: AppearanceRatingModel) => { model.evaluation.spearman = 1.1; },
    ]) {
      const model = fixture(); mutate(model);
      expect(() => parseAppearanceRatingModel(model)).toThrow();
      expect(predictAppearanceRating(input(), model, { allowResearchModel: true }).score).toBeNull();
    }
    expect(() => parseAppearanceRatingModel({ ...fixture(), researchOnly: false })).toThrow();
    expect(() => parseAppearanceRatingModel(null)).toThrow();
  });
  it("accepts the finite-sample boundary and copies imported coefficients", () => {
    const original = fixture(); original.calibration.subjects = 9;
    const parsed = parseAppearanceRatingModel(original);
    original.features[0]!.weight = 999;
    expect(parsed.features[0]!.weight).toBe(0.5);
  });
  it("treats nonfinite feature values as missing and never emits nonfinite ratings", () => {
    const result = predictAppearanceRating(input({ "legacy.canthalTilt": NaN, "legacy.fwhr": Infinity }), fixture(), { allowResearchModel: true });
    expect(result.status).toBe("unavailable"); expect(result.score).toBeNull();
  });
});

describe("optional strict display convention", () => {
  it("anchors the human-label reference center at five and is monotonic", () => {
    expect(strictDisplayRating(4, 4, 1)).toBeCloseTo(5);
    const scores = [-4, -2, 0, 2, 4].map(z => strictDisplayRating(4 + z, 4, 1));
    expect(scores).toEqual([...scores].sort((a, b) => a - b));
  });
  it("preserves an unreachable ceiling even under numerical saturation and formatting", () => {
    for (const x of [-1e6, 1e6]) { const score = strictDisplayRating(x, 4, 1); expect(score).toBeGreaterThan(1); expect(score).toBeLessThan(10); }
    expect(formatStrictDisplayRating(9.99)).toBe("9.9");
    expect(() => strictDisplayRating(4, 4, 0)).toThrow();
    expect(() => formatStrictDisplayRating(NaN)).toThrow();
  });
});
