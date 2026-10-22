import { describe, expect, it } from "vitest";
import { analyze } from "../src/index";
import { R_GONION, R_RAMUS } from "../src/landmarks/indices";
import type { ScanInput } from "../src/types";
import { canonicalInput } from "./helpers";

function expectRefused(input: ScanInput) {
  const result = analyze(input);
  expect(result.ok).toBe(false);
  expect(result.gates.pass).toBe(false);
  expect(result.gates.blocking.length).toBeGreaterThan(0);
  expect(result.metrics).toEqual([]);
  expect(result.overall).toBeNull();
  expect(result.overallPercentile).toBeNull();
  expect(result.standardized).toBeNull();
}

describe("untrusted landmark and capture input", () => {
  it.each([NaN, Infinity, -Infinity])("refuses a nonfinite coordinate (%s) instead of emitting a rating", (value) => {
    const input = canonicalInput();
    input.landmarks[152]!.x = value;
    expectRefused(input);
  });

  it("refuses a sparse mesh even when its array length is correct", () => {
    const input = canonicalInput();
    delete input.landmarks[152];
    expectRefused(input);
  });

  it.each([0, -1, NaN, Infinity, 1024.5])("refuses invalid image dimensions (%s)", (value) => {
    expectRefused(canonicalInput({ imageWidth: value }));
    expectRefused(canonicalInput({ imageHeight: value }));
  });

  it("refuses a nonfinite correction before it reaches geometric calculations", () => {
    expectRefused(canonicalInput({ overrides: { 152: { x: NaN, y: 0.5 } } }));
  });

  it("refuses a correction for a point outside the supplied topology", () => {
    expectRefused(canonicalInput({ overrides: { 999: { x: 0.5, y: 0.5 } } }));
  });

  it("refuses an RGBA buffer whose dimensions do not match the capture", () => {
    expectRefused(canonicalInput({ image: { width: 32, height: 32, data: new Uint8ClampedArray(32 * 32 * 4) } }));
  });

  it("refuses a truncated RGBA buffer instead of inventing black pixels", () => {
    expectRefused(canonicalInput({ image: { width: 1024, height: 1280, data: new Uint8ClampedArray(4) } }));
  });

  it("refuses nonfinite expression evidence instead of treating it as neutral", () => {
    expectRefused(canonicalInput({ blendshapes: { jawOpen: NaN } }));
  });

  it("does not mutate the caller's raw landmarks or manual corrections", () => {
    const input = canonicalInput({ overrides: { 152: { x: 0.5, y: 0.8 } } });
    const snapshot = JSON.stringify(input);
    analyze(input);
    expect(JSON.stringify(input)).toBe(snapshot);
  });

  it("treats a coincident jaw tangent as unavailable rather than a measured zero-degree angle", () => {
    const input = canonicalInput();
    input.landmarks[R_RAMUS] = { ...input.landmarks[R_GONION]! };
    const result = analyze(input);
    expect(result.metrics.some(metric => metric.key === "jawAngularity")).toBe(false);
    expect(result.metrics.every(metric => Number.isFinite(metric.value) && Number.isFinite(metric.score))).toBe(true);
    expect(result.overall === null || Number.isFinite(result.overall)).toBe(true);
  });

  it("preserves measurements when the same pixels are padded to a different image aspect ratio", () => {
    const input = canonicalInput();
    const width = 2560, height = 1440;
    const offsetX = (width - input.imageWidth) / 2;
    const offsetY = (height - input.imageHeight) / 2;
    const padded = analyze({
      ...input,
      imageWidth: width,
      imageHeight: height,
      landmarks: input.landmarks.map(point => ({
        x: (point.x * input.imageWidth + offsetX) / width,
        y: (point.y * input.imageHeight + offsetY) / height,
        z: point.z * input.imageWidth / width,
      })),
    });
    const original = analyze(input);
    expect(padded.ok).toBe(true);
    expect(padded.overall).toBe(original.overall);
    for (const metric of original.metrics) {
      expect(padded.metrics.find(other => other.key === metric.key)?.value).toBeCloseTo(metric.value, 8);
    }
  });
});
