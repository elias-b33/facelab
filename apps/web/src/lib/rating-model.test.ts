import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { ENGINE_VERSION, RATING_FEATURE_KEYS, ratingModelReadiness, type RatingModel } from "@freeharmony/engine";
import {
  MAX_RATING_MODEL_BYTES,
  RATING_MODEL_EVENT,
  RATING_MODEL_KEY,
  readRatingModel,
  removeRatingModel,
  saveRatingModel,
} from "./rating-model";

/** Explicitly synthetic protocol fixture, never eligible to rate real people. */
function syntheticModel(): RatingModel {
  return {
    schemaVersion: 1, kind: "facelab-ridge-rating", featureSchema: "facelab-frontal-geometry-v1",
    modelId: "storage-unit-test-only", createdAt: "2026-10-22T00:00:00Z",
    target: { label: "Synthetic test target", min: 1, max: 5, population: "Generated numbers only", protocol: "Software test; no human ratings" },
    provenance: {
      datasetId: "synthetic-storage-test", datasetName: "Synthetic storage test fixture",
      sourceUrl: "https://example.invalid/never-fetch-this", license: "Test fixture only", usage: "research",
      identityProtocol: "Generated identifiers only", dataSha256: "a".repeat(64), splitSeed: "test",
      engineVersion: ENGINE_VERSION, synthetic: true,
    },
    features: RATING_FEATURE_KEYS.map(key => ({ key, mean: 0, scale: 1, linear: 0, quadratic: 0, min: -1, max: 1 })),
    intercept: 3,
    calibration: { method: "split-conformal-absolute-residual", coverage: .9, absoluteErrorQuantile: .3, subjects: 100 },
    evaluation: { testSubjects: 100, testRows: 100, testTotalRows: 100, testUnusableRows: 0, testCoverage: 1,
      mae: .2, rmse: .25, pearson: .7, spearman: .7, baselineMae: .5, baselineRmse: .6,
      intervalCoverage: .9, meanIntervalWidth: .6 },
    quality: { minimumFeatureConfidence: .5, maxAbsZ: 6 },
    training: { subjects: 300, validationSubjects: 100, regularization: .1, basis: "linear" },
  };
}

let values: Map<string, string>;
let setItem: ReturnType<typeof vi.fn>;
let dispatchEvent: ReturnType<typeof vi.fn>;
let fetch: ReturnType<typeof vi.fn>;

beforeEach(() => {
  values = new Map();
  setItem = vi.fn((key: string, value: string) => { values.set(key, value); });
  dispatchEvent = vi.fn();
  fetch = vi.fn();
  vi.stubGlobal("window", { dispatchEvent });
  vi.stubGlobal("fetch", fetch);
  vi.stubGlobal("localStorage", {
    getItem: vi.fn((key: string) => values.get(key) ?? null),
    setItem,
    removeItem: vi.fn((key: string) => { values.delete(key); }),
  });
});
afterEach(() => { vi.unstubAllGlobals(); });

describe("local rating-model storage", () => {
  it("round-trips data locally and retains the synthetic-model refusal", () => {
    const model = saveRatingModel(JSON.stringify(syntheticModel()));
    const restored = readRatingModel();
    expect(restored).toEqual({ model, error: null });
    expect(ratingModelReadiness(restored.model!).some(message => message.includes("Synthetic"))).toBe(true);
    expect(dispatchEvent).toHaveBeenCalledWith(expect.objectContaining({ type: RATING_MODEL_EVENT }));
    expect(fetch).not.toHaveBeenCalled();
  });

  it("does not replace a working model with corrupt JSON or an invalid artifact", () => {
    saveRatingModel(JSON.stringify(syntheticModel()));
    const previous = values.get(RATING_MODEL_KEY);
    for (const bad of ["{", JSON.stringify({ ...syntheticModel(), features: [] })]) {
      expect(() => saveRatingModel(bad)).toThrow();
      expect(values.get(RATING_MODEL_KEY)).toBe(previous);
    }
    expect(fetch).not.toHaveBeenCalled();
  });

  it("enforces the size limit in bytes for both ASCII and multibyte JSON", () => {
    saveRatingModel(JSON.stringify(syntheticModel()));
    const previous = values.get(RATING_MODEL_KEY);
    const oversized = [
      JSON.stringify({ ...syntheticModel(), padding: "x".repeat(MAX_RATING_MODEL_BYTES) }),
      JSON.stringify({ ...syntheticModel(), padding: "å".repeat(MAX_RATING_MODEL_BYTES / 2) }),
    ];
    for (const raw of oversized) {
      expect(new TextEncoder().encode(raw).byteLength).toBeGreaterThan(MAX_RATING_MODEL_BYTES);
      expect(() => saveRatingModel(raw)).toThrow(/256 KB/);
      expect(values.get(RATING_MODEL_KEY)).toBe(previous);
    }
  });

  it("reports oversized saved data without deleting it or requesting a URL", () => {
    const raw = JSON.stringify({ ...syntheticModel(), padding: "å".repeat(MAX_RATING_MODEL_BYTES / 2) });
    values.set(RATING_MODEL_KEY, raw);
    const restored = readRatingModel();
    expect(restored.model).toBeNull();
    expect(restored.error).toContain("supported size");
    expect(values.get(RATING_MODEL_KEY)).toBe(raw);
    expect(fetch).not.toHaveBeenCalled();
  });

  it("keeps the previous model when the browser rejects a storage write", () => {
    saveRatingModel(JSON.stringify(syntheticModel()));
    const previous = values.get(RATING_MODEL_KEY);
    dispatchEvent.mockClear();
    setItem.mockImplementation(() => { throw new Error("QuotaExceededError"); });
    expect(() => saveRatingModel(JSON.stringify({ ...syntheticModel(), modelId: "replacement-test" }))).toThrow(/previous model has been kept/);
    expect(values.get(RATING_MODEL_KEY)).toBe(previous);
    expect(dispatchEvent).not.toHaveBeenCalled();
  });

  it("removes only the model and keeps the user's stored studies", () => {
    saveRatingModel(JSON.stringify(syntheticModel()));
    values.set("faceiq.scans.v1", "stored-studies");
    removeRatingModel();
    expect(values.has(RATING_MODEL_KEY)).toBe(false);
    expect(values.get("faceiq.scans.v1")).toBe("stored-studies");
    expect(readRatingModel()).toEqual({ model: null, error: null });
  });

  it("can be read during server rendering without accessing browser storage", () => {
    vi.stubGlobal("window", undefined);
    expect(readRatingModel()).toEqual({ model: null, error: null });
  });
});
