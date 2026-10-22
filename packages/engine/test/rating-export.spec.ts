import { afterAll, beforeAll, describe, expect, it } from "vitest";
import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { convertRatingRows } from "../scripts/export-rating-features";
import { ENGINE_VERSION, RATING_FEATURE_KEYS } from "../src/index";
import { canonicalIrisInput } from "./helpers";

let sourceDirectory: string;

beforeAll(() => {
  sourceDirectory = mkdtempSync(join(tmpdir(), "facelab-rating-export-test-"));
  const input = canonicalIrisInput();
  const pixels = new Uint8Array(input.imageWidth * input.imageHeight * 4);
  // Synthetic sharp, exposed pixels; no user photos or personal data.
  for (let y = 0; y < input.imageHeight; y++) for (let x = 0; x < input.imageWidth; x++) {
    const offset = (y * input.imageWidth + x) * 4;
    const shade = 70 + ((x + y) % 7) * 20;
    pixels[offset] = pixels[offset + 1] = pixels[offset + 2] = shade;
    pixels[offset + 3] = 255;
  }
  writeFileSync(join(sourceDirectory, "synthetic.rgba"), pixels);
  writeFileSync(join(sourceDirectory, "truncated.rgba"), new Uint8Array(4));
});

afterAll(() => {
  if (sourceDirectory) rmSync(sourceDirectory, { recursive: true, force: true });
});

function sourceRow(id = "photo-a", subject = "person-a"): Record<string, unknown> {
  const input = canonicalIrisInput();
  return {
    id, subject_id: subject, identity_verified: true, view: "front",
    w: input.imageWidth, h: input.imageHeight,
    landmarks: input.landmarks.map(point => [point.x, point.y, point.z]),
    matrix: input.transformationMatrix,
    rgba_path: "synthetic.rgba",
    rating: { mean: 3, count: 10, scale: [1, 5] },
  };
}

describe("rating feature export uses exact identity and label records", () => {
  it("joins labels by exact photo ID, not row order or an embedded fallback", () => {
    const labels = [
      { id: "photo-b", rating: { mean: 4.1, count: 8, scale: [1, 5] } },
      { id: "photo-a", rating: { mean: 1.9, count: 12, scale: [1, 5] } },
    ];
    const rows = convertRatingRows([sourceRow(), sourceRow("photo-b", "person-b")], labels, sourceDirectory);
    expect(rows.map(row => row.id)).toEqual(["photo-a", "photo-b"]);
    expect(rows.map(row => row.rating)).toEqual([labels[1]!.rating, labels[0]!.rating]);
    expect(rows.every(row => row.quality_pass)).toBe(true);
    expect(rows[0]!.engine_version).toBe(ENGINE_VERSION);
    expect(Object.keys(rows[0]!.features)).toEqual([...RATING_FEATURE_KEYS]);
  });

  it("rejects missing and unmatched labels rather than matching approximately", () => {
    expect(() => convertRatingRows([sourceRow()], [
      { id: "PHOTO-A", rating: { mean: 3, count: 10, scale: [1, 5] } },
    ], sourceDirectory)).toThrow(/exact joined human rating/);
    expect(() => convertRatingRows([sourceRow()], [
      { id: "photo-a", rating: { mean: 3, count: 10, scale: [1, 5] } },
      { id: "unused-photo", rating: { mean: 3, count: 10, scale: [1, 5] } },
    ], sourceDirectory)).toThrow(/no matching landmark row/);
  });

  it("rejects duplicate photo IDs, duplicate subject IDs, and duplicate label IDs", () => {
    expect(() => convertRatingRows([sourceRow(), sourceRow("photo-a", "person-b")], [], sourceDirectory)).toThrow(/unique/);
    expect(() => convertRatingRows([sourceRow(), sourceRow("photo-b", "person-a")], [], sourceDirectory)).toThrow(/one preselected photo/);
    expect(() => convertRatingRows([sourceRow()], [
      { id: "photo-a", rating: { mean: 3, count: 10, scale: [1, 5] } },
      { id: "photo-a", rating: { mean: 4, count: 10, scale: [1, 5] } },
    ], sourceDirectory)).toThrow(/unique/);
  });

  it("requires an explicit verified person identity", () => {
    expect(() => convertRatingRows([{ ...sourceRow(), identity_verified: false }], [], sourceDirectory)).toThrow(/verified person identity/);
  });
});

describe("rating feature export preserves abstentions", () => {
  it("retains geometry without pixels but marks it ineligible for supervised fitting", () => {
    const row = sourceRow();
    delete row.rgba_path;
    const [exported] = convertRatingRows([row], [], sourceDirectory);
    expect(exported!.quality_pass).toBe(false);
    expect(exported!.features.canthalTilt).toBeCloseTo(1.74, 1);
    expect(Object.keys(exported!.features)).toHaveLength(RATING_FEATURE_KEYS.length);
    expect(exported!.issues.join(" ")).toContain("Photo-quality pixels are absent");
    expect(exported!.rating).toEqual(row.rating);
  });

  it("fails on mismatched pixel bytes instead of assuming absent bytes are valid pixels", () => {
    expect(() => convertRatingRows([{ ...sourceRow(), rgba_path: "truncated.rgba" }], [], sourceDirectory)).toThrow(/RGBA bytes do not match/);
  });

  it("keeps a no-face row and its human label in the evaluation denominator", () => {
    const rows = convertRatingRows([
      sourceRow(),
      { ...sourceRow("failed-photo", "failed-person"), landmarks: [], detection_issue: "No face was detected in the source photograph." },
    ], [], sourceDirectory);
    expect(rows).toHaveLength(2);
    expect(rows.filter(row => row.quality_pass)).toHaveLength(1);
    expect(rows[1]).toMatchObject({ id: "failed-photo", subject_id: "failed-person", quality_pass: false, features: {}, confidence: {} });
    expect(rows[1]!.rating).toEqual({ mean: 3, count: 10, scale: [1, 5] });
    expect(rows[1]!.issues.join(" ")).toContain("No face");
  });

  it("preserves an explicit detection problem even when a landmark array exists", () => {
    const [row] = convertRatingRows([{ ...sourceRow(), detection_issue: "Multiple faces were present." }], [], sourceDirectory);
    expect(row!.quality_pass).toBe(false);
    expect(row!.issues).toContain("Multiple faces were present.");
  });
});
