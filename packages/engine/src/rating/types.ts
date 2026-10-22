import type { MetricKey } from "../types";

/** A learned prediction of a named panel's mean rating, on its original scale. */
export interface RatingModel {
  schemaVersion: 1;
  kind: "facelab-ridge-rating";
  featureSchema: "facelab-frontal-geometry-v1";
  modelId: string;
  createdAt: string;
  target: { label: string; min: number; max: number; population: string; protocol: string };
  provenance: {
    datasetId: string;
    datasetName: string;
    sourceUrl: string;
    license: string;
    usage: "research" | "commercial-permitted";
    identityProtocol: string;
    dataSha256: string;
    splitSeed: string;
    engineVersion: string;
    synthetic: boolean;
  };
  features: {
    key: MetricKey;
    mean: number;
    scale: number;
    linear: number;
    quadratic: number;
    min: number;
    max: number;
  }[];
  intercept: number;
  calibration: {
    method: "split-conformal-absolute-residual";
    coverage: number;
    absoluteErrorQuantile: number;
    subjects: number;
  };
  evaluation: {
    testSubjects: number;
    testRows: number;
    testTotalRows: number;
    testUnusableRows: number;
    testCoverage: number;
    mae: number;
    rmse: number;
    pearson: number | null;
    spearman: number | null;
    baselineMae: number;
    baselineRmse: number;
    intervalCoverage: number;
    meanIntervalWidth: number;
  };
  quality: { minimumFeatureConfidence: number; maxAbsZ: number };
  training: { subjects: number; validationSubjects: number; regularization: number; basis: "linear" | "quadratic" };
}

export interface RatingFeatures {
  feature_schema: "facelab-frontal-geometry-v1";
  engine_version: string;
  quality_pass: boolean;
  features: Partial<Record<MetricKey, number>>;
  confidence: Partial<Record<MetricKey, number>>;
  issues: string[];
}

export type RatingPrediction = {
  status: "unavailable" | "withheld";
  value: null;
  reason: "no-model" | "invalid-model" | "synthetic-model" | "unvalidated-model" | "engine-mismatch" | "photo-quality" | "missing-features" | "outside-training-range";
  messages: string[];
} | {
  status: "predicted";
  value: number;
  modelId: string;
  target: RatingModel["target"];
  interval: { low: number; high: number; coverage: number };
  evaluation: RatingModel["evaluation"];
  /** Local model terms relative to the training mean; not causal advice. */
  contributions: { key: MetricKey; value: number }[];
  messages: string[];
};

export type RatingModelValidation = { valid: true; model: RatingModel; errors: [] } | { valid: false; model: null; errors: string[] };
