import { describe, expect, it } from "vitest";
import { aggregate } from "../src/scoring/aggregate";
import { overallPercentileOf, percentileOf, standardizedOverall } from "../src/scoring/norms";
import type { AreaKey, MetricKey, MetricResult } from "../src/types";

function metric(key: MetricKey, score: number, confidence = 1): MetricResult {
  return { key, label: key, value: 1, unit: "ratio", band: { lo: 0, hi: 2 }, score, verdict: "near-ideal", confidence, flags: [], percentile: null };
}

const weights: Record<string, { area: AreaKey; weight: number }> = {
  overallSymmetry: { area: "symmetry", weight: 1 },
  canthalTilt: { area: "eyeArea", weight: 1 },
  midfaceRatio: { area: "midface", weight: 1 },
  jawToCheekbone: { area: "jawline", weight: 1 },
};
const baseline = () => [
  metric("overallSymmetry", 80),
  metric("canthalTilt", 70),
  metric("midfaceRatio", 60),
  metric("jawToCheekbone", 50),
];

describe("score aggregation with incomplete evidence", () => {
  it("a uniform confidence reduction changes confidence, not the geometry score", () => {
    const original = aggregate(baseline(), weights);
    const lessCertain = aggregate(baseline().map(m => ({ ...m, confidence: 0.8 })), weights);
    expect(lessCertain.overall).toBe(original.overall);
    for (const area of Object.keys(weights).map(key => weights[key]!.area)) {
      expect(lessCertain.areas[area].score).toBe(original.areas[area].score);
      expect(lessCertain.areas[area].confidence).toBeCloseTo(0.8, 10);
    }
  });

  it("a missing high-weight measure reduces area coverage instead of increasing certainty", () => {
    const withExpectedMeasure = {
      ...weights,
      eyeSeparationRatio: { area: "eyeArea" as const, weight: 9 },
    };
    const result = aggregate(baseline(), withExpectedMeasure);
    expect(result.areas.eyeArea.confidence).toBeCloseTo(0.1, 10);
    expect(result.areas.eyeArea.score).toBeNull();
  });

  it.each(["score", "confidence", "value"] as const)("nonfinite %s cannot contaminate the aggregate", (field) => {
    for (const invalid of [NaN, Infinity, -Infinity]) {
      const items = baseline();
      items[1] = { ...items[1]!, [field]: invalid };
      const result = aggregate(items, weights);
      expect(result.areas.eyeArea.score).toBeNull();
      expect(result.areas.eyeArea.confidence).toBe(0);
      expect(result.overall === null || Number.isFinite(result.overall)).toBe(true);
    }
  });

  it("refuses an aggregate when two whole areas are unavailable", () => {
    const result = aggregate(baseline().slice(0, 2), weights);
    expect(result.overall).toBeNull();
    expect(result.tier).toBeNull();
  });
});

describe("norm lookups are not a fallback for invalid numbers", () => {
  it.each([NaN, Infinity, -Infinity])("returns null for nonfinite values (%s)", (value) => {
    expect(percentileOf("canthalTilt", value)).toBeNull();
    expect(overallPercentileOf(value)).toBeNull();
    expect(standardizedOverall(value)).toBeNull();
  });
});
