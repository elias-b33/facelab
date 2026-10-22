import type { Pt } from "./types";
import { L_TRAGION, PRONASALE, R_TRAGION } from "./landmarks/indices";

export interface PoseEstimate {
  yawDeg: number | null;
  pitchDeg: number | null;
  rollDeg: number | null;
  source: "matrix" | "landmark-proxy";
  yawAsym: number;
}

/**
 * Detect whether a flat 16-float 4x4 is column-major or row-major.
 * In a rigid transform exactly one of the two candidate off-diagonal triples
 * is ~0 (the perspective row) while the other carries the translation.
 * With zero translation the layout is ambiguous; retain the row-major
 * convention for that case. A camera-relative MediaPipe pose has translation.
 */
export function isColumnMajor(d: number[]): boolean {
  const a = Math.abs(d[3]!) + Math.abs(d[7]!) + Math.abs(d[11]!);
  const b = Math.abs(d[12]!) + Math.abs(d[13]!) + Math.abs(d[14]!);
  return a < b;
}

type Mat3 = number[][];

function rotationFromFlat(d: number[]): Mat3 {
  // Produce R[row][col] regardless of storage order.
  if (isColumnMajor(d)) {
    return [
      [d[0]!, d[4]!, d[8]!],
      [d[1]!, d[5]!, d[9]!],
      [d[2]!, d[6]!, d[10]!],
    ];
  }
  return [
    [d[0]!, d[1]!, d[2]!],
    [d[4]!, d[5]!, d[6]!],
    [d[8]!, d[9]!, d[10]!],
  ];
}

/** A face pose is an affine proper rotation, optionally with uniform scale.
 * Zero, non-finite, perspective, reflection and shearing matrices must not
 * turn into a confident "straight-on" pose through atan2(0, 0). */
export function isValidPoseMatrix(d: number[] | undefined): d is number[] {
  if (!Array.isArray(d) || d.length !== 16) return false;
  for (let i = 0; i < 16; i++) if (!Number.isFinite(d[i])) return false;
  if (Math.abs(d[15]! - 1) > 1e-4) return false;
  const perspective = isColumnMajor(d) ? [d[3]!, d[7]!, d[11]!] : [d[12]!, d[13]!, d[14]!];
  if (perspective.some(value => Math.abs(value) > 1e-4)) return false;
  const r = rotationFromFlat(d);
  const norms = r.map(row => Math.hypot(...row));
  const scale = (norms[0]! + norms[1]! + norms[2]!) / 3;
  if (!Number.isFinite(scale) || scale < 1e-6 || norms.some(n => Math.abs(n / scale - 1) > .02)) return false;
  const normalized = r.map(row => row.map(v => v / scale));
  for (let a = 0; a < 3; a++) for (let b = a + 1; b < 3; b++) {
    const dot = normalized[a]!.reduce((sum, value, i) => sum + value * normalized[b]![i]!, 0);
    if (Math.abs(dot) > .02) return false;
  }
  const [a, b, c] = normalized as [number[], number[], number[]];
  const determinant = a[0]! * (b[1]! * c[2]! - b[2]! * c[1]!) -
    a[1]! * (b[0]! * c[2]! - b[2]! * c[0]!) + a[2]! * (b[0]! * c[1]! - b[1]! * c[0]!);
  return Math.abs(determinant - 1) <= .03;
}

const RAD2DEG = 180 / Math.PI;

/**
 * Factor R = Rz(roll)·Ry(yaw)·Rx(pitch). Sign conventions are pinned by the
 * synthetic-rotation test in test/pose.spec.ts — do not "fix" them from docs.
 */
export function eulerFromMatrix(d: number[]): {
  pitchDeg: number;
  yawDeg: number;
  rollDeg: number;
} {
  if (!isValidPoseMatrix(d)) throw new RangeError("The face pose matrix is not a finite rigid transformation.");
  const r = rotationFromFlat(d);
  const pitch = Math.atan2(r[2]![1]!, r[2]![2]!);
  const yaw = Math.atan2(
    -r[2]![0]!,
    Math.sqrt(r[2]![1]! * r[2]![1]! + r[2]![2]! * r[2]![2]!),
  );
  const roll = Math.atan2(r[1]![0]!, r[0]![0]!);
  return {
    pitchDeg: pitch * RAD2DEG,
    yawDeg: yaw * RAD2DEG,
    rollDeg: roll * RAD2DEG,
  };
}

/**
 * Landmark yaw proxy: nose-tip displacement projected onto the cheek-to-cheek
 * axis. Projection, unlike screen-x distances, is invariant to in-plane roll.
 * Natural asymmetry can move this proxy; it is not an anatomical yaw measure
 * and must never overrule a valid pose matrix.
 */
export function yawAsymmetry(pts: Pt[]): number {
  const nose = pts[PRONASALE];
  const tr = pts[R_TRAGION];
  const tl = pts[L_TRAGION];
  if (!nose || !tr || !tl || [nose, tr, tl].some(p => !Number.isFinite(p.x) || !Number.isFinite(p.y))) return 0;
  const ax = tl.x - tr.x, ay = tl.y - tr.y;
  const width = Math.hypot(ax, ay);
  if (!Number.isFinite(width) || width <= 1e-8) return 0;
  const projected = ((nose.x - tr.x) * ax + (nose.y - tr.y) * ay) / width;
  const dR = Math.abs(projected);
  const dL = Math.abs(width - projected);
  const sum = dL + dR;
  if (sum === 0) return 0;
  return (dL - dR) / sum;
}

/**
 * Proxy→degrees factor, measured numerically on the canonical model
 * (yaw 2–12° maps at −43.2…−43.8 deg per unit asymmetry; see pose tests).
 * Negative: a positive geometric yaw pushes the nose toward the subject's
 * right tragion, making (dL − dR) negative. Sign convention against real
 * MediaPipe matrices is re-validated in the browser adapter's smoke test.
 */
export const YAW_PROXY_TO_DEG = -43.7;

export function estimatePose(
  pts: Pt[],
  transformationMatrix: number[] | undefined,
): PoseEstimate {
  const yawAsym = yawAsymmetry(pts);
  if (isValidPoseMatrix(transformationMatrix)) {
    const e = eulerFromMatrix(transformationMatrix);
    return {
      yawDeg: e.yawDeg,
      pitchDeg: e.pitchDeg,
      rollDeg: e.rollDeg,
      source: "matrix",
      yawAsym,
    };
  }
  return {
    yawDeg: yawAsym * YAW_PROXY_TO_DEG,
    pitchDeg: null,
    rollDeg: null,
    source: "landmark-proxy",
    yawAsym,
  };
}
