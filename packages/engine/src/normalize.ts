import type { Frame, Gate, Pt, ScanInput } from "./types";
import { estimatePose, type PoseEstimate } from "./pose";
import {
  L_CANTHUS_LAT,
  L_CANTHUS_MED,
  L_IRIS_C,
  R_CANTHUS_LAT,
  R_CANTHUS_MED,
  R_IRIS_C,
} from "./landmarks/indices";

const normalizedPoint = (p: { x: number; y: number } | undefined): boolean =>
  !!p && Number.isFinite(p.x) && Number.isFinite(p.y) &&
  p.x >= 0 && p.x <= 1 && p.y >= 0 && p.y <= 1;

/** Validate untrusted model/imported inputs before any ratios or image allocation.
 * A malformed optional pose matrix is deliberately not a fatal input error:
 * estimatePose() can still return the explicitly labelled landmark proxy. */
export function validateScanInput(input: ScanInput): Gate[] {
  const problems: Gate[] = [];
  const fail = (code: string, message: string) => problems.push({
    code, severity: "block", message,
    retake: "Use a clear photo with the whole face visible, then run detection again.",
  });
  if (!input || !Array.isArray(input.landmarks) ||
      (input.landmarks.length !== 468 && input.landmarks.length !== 478)) {
    fail("no-face", "A complete face landmark set was not found.");
    return problems;
  }
  const { imageWidth: w, imageHeight: h } = input;
  if ((input.sex !== undefined && !["neutral", "masculine", "feminine"].includes(input.sex)) ||
      (input.bandProfile !== undefined && !["faceharmony-parity", "literature", "calibrated"].includes(input.bandProfile))) {
    fail("invalid-settings", "The selected measurement settings are invalid.");
  }
  if (!Number.isSafeInteger(w) || !Number.isSafeInteger(h) || w <= 0 || h <= 0) {
    fail("invalid-image", "The photo dimensions are invalid.");
  }
  // Do not use Array.every: it skips holes in sparse imported arrays.
  for (let i = 0; i < input.landmarks.length; i++) {
    const p = input.landmarks[i];
    if (!normalizedPoint(p) || !Number.isFinite(p?.z) || !Number.isFinite(p!.z * w)) {
      fail("invalid-landmarks", "The landmark data are incomplete, non-finite, or outside the photo.");
      break;
    }
  }
  if (input.overrides !== undefined) {
    if (!input.overrides || typeof input.overrides !== "object" || Array.isArray(input.overrides)) {
      fail("invalid-overrides", "The adjusted landmark data are invalid.");
    } else for (const [key, p] of Object.entries(input.overrides)) {
      const index = Number(key);
      if (!Number.isInteger(index) || String(index) !== key || index < 0 || index >= input.landmarks.length || !normalizedPoint(p)) {
        fail("invalid-overrides", "An adjusted landmark is invalid or outside the photo.");
        break;
      }
    }
  }
  if (input.blendshapes !== undefined && (!input.blendshapes ||
      typeof input.blendshapes !== "object" || Array.isArray(input.blendshapes) ||
      Object.values(input.blendshapes).some(v => !Number.isFinite(v) || v < 0 || v > 1))) {
    fail("invalid-blendshapes", "The expression model returned invalid values.");
  }
  if (input.image !== undefined) {
    const image = input.image;
    const pixelData: unknown = image?.data;
    const bytes = w * h * 4;
    if (!image || image.width !== w || image.height !== h || !Number.isSafeInteger(bytes) ||
        !(pixelData instanceof Uint8ClampedArray || pixelData instanceof Uint8Array) || pixelData.length !== bytes) {
      fail("invalid-image", "The photo pixels do not match the landmark image dimensions.");
    }
  }
  if (problems.length === 0) {
    const point = (i: number) => input.overrides?.[i] ?? input.landmarks[i]!;
    const center = (a: number, b: number) => ({ x: (point(a).x + point(b).x) / 2, y: (point(a).y + point(b).y) / 2 });
    const right = input.landmarks.length === 478 ? point(R_IRIS_C) : center(R_CANTHUS_LAT, R_CANTHUS_MED);
    const left = input.landmarks.length === 478 ? point(L_IRIS_C) : center(L_CANTHUS_MED, L_CANTHUS_LAT);
    const separation = Math.hypot((left.x - right.x) * w, (left.y - right.y) * h);
    if (!Number.isFinite(separation) || separation <= 1e-8) {
      fail("degenerate-landmarks", "The eye reference points coincide, so facial ratios cannot be measured.");
    }
  }
  return problems;
}

/**
 * Build the measurement frame from raw MediaPipe output. Ordered steps:
 *   0. de-mirror            (before everything — else anatomical L/R swap)
 *   1. Adjust Points overrides
 *   2. de-normalize to px   (before any angle math — normalized coords are
 *                            anisotropic on non-square images)
 *   3. flip y-up
 *   4. roll-correct about the pupil midpoint
 *   5. scale = post-rotation interpupillary distance
 */
export function buildFrame(input: ScanInput): { frame: Frame; pose: PoseEstimate } {
  const errors = validateScanInput(input);
  if (errors.length) throw new RangeError(errors.map(error => error.message).join(" "));
  const { imageWidth: w, imageHeight: h } = input;
  const hasIris = input.landmarks.length >= 478;

  // Steps 0–3
  const pts: Pt[] = new Array(input.landmarks.length);
  for (let i = 0; i < input.landmarks.length; i++) {
    const raw = input.landmarks[i]!;
    const o = input.overrides?.[i];
    let nx = o ? o.x : raw.x;
    let ny = o ? o.y : raw.y;
    const nz = raw.z;
    if (input.mirrored) nx = 1 - nx;
    pts[i] = { x: nx * w, y: h - ny * h, z: nz * w };
  }

  // Pose is estimated on the pre-roll-corrected pixel points (the transform
  // matrix, when present, already encodes roll).
  const pose = estimatePose(pts, input.transformationMatrix);

  // Step 4 — roll correction about the pupil midpoint.
  const pr = hasIris ? pts[R_IRIS_C]! : midOf(pts, R_CANTHUS_LAT, R_CANTHUS_MED);
  const pl = hasIris ? pts[L_IRIS_C]! : midOf(pts, L_CANTHUS_MED, L_CANTHUS_LAT);
  const theta = Math.atan2(pl.y - pr.y, pl.x - pr.x);
  const cx = (pr.x + pl.x) / 2;
  const cy = (pr.y + pl.y) / 2;
  const c = Math.cos(-theta);
  const s = Math.sin(-theta);
  for (let i = 0; i < pts.length; i++) {
    const p = pts[i]!;
    const ux = p.x - cx;
    const uy = p.y - cy;
    pts[i] = { x: cx + ux * c - uy * s, y: cy + ux * s + uy * c, z: p.z };
  }

  // Step 5 — scale (denominator only; points stay in pixels for image gates).
  const pr2 = hasIris ? pts[R_IRIS_C]! : midOf(pts, R_CANTHUS_LAT, R_CANTHUS_MED);
  const pl2 = hasIris ? pts[L_IRIS_C]! : midOf(pts, L_CANTHUS_MED, L_CANTHUS_LAT);
  const ddx = pl2.x - pr2.x;
  const ddy = pl2.y - pr2.y;
  const scale = Math.sqrt(ddx * ddx + ddy * ddy);

  const frame: Frame = {
    pts,
    scale,
    rollDeg: (theta * 180) / Math.PI,
    yawDeg: pose.yawDeg,
    pitchDeg: pose.pitchDeg,
    poseSource: pose.source,
    yawAsym: pose.yawAsym,
    imageWidth: w,
    imageHeight: h,
    hasIris,
  };
  return { frame, pose };
}

/** Coordinates for sampling the original pixels, with no roll correction or
 * de-mirroring. Geometry uses buildFrame(); image quality must use this frame.
 * In particular, a mirrored photograph still has mirrored pixels. */
export function buildImageFrame(input: ScanInput, frame: Frame): Frame {
  return {
    ...frame,
    rollDeg: 0,
    pts: input.landmarks.map((raw, i) => {
      const p = input.overrides?.[i] ?? raw;
      return { x: p.x * input.imageWidth, y: (1 - p.y) * input.imageHeight, z: raw.z * input.imageWidth };
    }),
  };
}

function midOf(pts: Pt[], a: number, b: number): Pt {
  const pa = pts[a]!;
  const pb = pts[b]!;
  return { x: (pa.x + pb.x) / 2, y: (pa.y + pb.y) / 2, z: (pa.z + pb.z) / 2 };
}
