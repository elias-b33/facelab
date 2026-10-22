import type {
  FaceRegion,
  Frame,
  Gate,
  GateReport,
  ScanInput,
} from "../types";
import {
  FACE_OVAL,
  L_CANTHUS_LAT,
  L_CANTHUS_MED,
  L_LID_INF,
  L_LID_INF2,
  L_LID_SUP,
  L_LID_SUP2,
  R_CANTHUS_LAT,
  R_CANTHUS_MED,
  R_LID_INF,
  R_LID_INF2,
  R_LID_SUP,
  R_LID_SUP2,
} from "../landmarks/indices";
import { dx, dy, pt } from "../landmarks/accessors";
import {
  boxDownsample,
  clampRect,
  cropGray,
  toGray,
  type GrayPlane,
} from "../image/ops";
import { regionalResiduals } from "../procrustes";
import { computeJawEdgeSupport } from "../metrics/registry";
import { buildImageFrame } from "../normalize";

// Thresholds. Values marked CALIBRATE are first-guess constants to be
// re-anchored with the calibrate harness on a real photo corpus.
const MIN_IPD_PX = 80;
const DEGRADE_IPD_PX = 140;
const MIN_FRAME_PX = 720;
const YAW_FULL = 6;
const YAW_DEGRADE = 12;
const PITCH_FULL = 6;
const PITCH_DEGRADE = 10;
const ROLL_DEGRADE = 12;
const ROLL_REFUSE = 22;
// Proxy magnitudes measured on the canonical model at 6° / 12° of yaw.
const YAW_PROXY_FULL = 0.137;
const YAW_PROXY_DEGRADE = 0.278;
const BLUR_REJECT = 0.008; // CALIBRATE
const BLUR_DEGRADE = 0.018; // CALIBRATE
const CLIP_REJECT = 0.12;
const CLIP_DEGRADE = 0.06;
const BLINK_REJECT = 0.55;
const EAR_SMALL = 0.15;
const EAR_ASYM_WARN = 0.07;
const JAW_OPEN_REJECT = 0.3;
const JAW_OPEN_WARN = 0.18;
const SMILE_REJECT = 0.45;
const SMILE_WARN = 0.28;
// Distance from one canonical shape cannot establish occlusion or accuracy.
// Keep this as an overlay-review diagnostic, without penalizing face shape.
const RESIDUAL_REVIEW = 0.16; // CALIBRATE
const JAW_EDGE_MIN = 0.25; // CALIBRATE

function gate(
  code: string,
  severity: "block" | "warn",
  message: string,
  retake: string,
): Gate {
  return { code, severity, message, retake };
}

/** Eye aspect ratio; canonical open eye ≈ 0.254, closed ≈ 0.08–0.12. */
export function eyeAspectRatio(f: Frame, side: "right" | "left"): number {
  if (side === "right") {
    const h1 = dy(f, R_LID_SUP, R_LID_INF);
    const h2 = dy(f, R_LID_SUP2, R_LID_INF2);
    return (h1 + h2) / (2 * dx(f, R_CANTHUS_LAT, R_CANTHUS_MED));
  }
  const h1 = dy(f, L_LID_SUP, L_LID_INF);
  const h2 = dy(f, L_LID_SUP2, L_LID_INF2);
  return (h1 + h2) / (2 * dx(f, L_CANTHUS_MED, L_CANTHUS_LAT));
}

function faceOvalBounds(f: Frame): { x: number; y: number; w: number; h: number } {
  let minX = Infinity;
  let minY = Infinity;
  let maxX = -Infinity;
  let maxY = -Infinity;
  for (const i of FACE_OVAL) {
    const p = pt(f, i);
    if (p.x < minX) minX = p.x;
    if (p.x > maxX) maxX = p.x;
    if (p.y < minY) minY = p.y;
    if (p.y > maxY) maxY = p.y;
  }
  // Frame is y-up; convert to image-space y-down for pixel cropping.
  const pad = 0.1;
  const w = (maxX - minX) * (1 + 2 * pad);
  const h = (maxY - minY) * (1 + 2 * pad);
  const x = minX - (maxX - minX) * pad;
  const yTopImage = f.imageHeight - (maxY + (maxY - minY) * pad);
  return { x, y: yTopImage, w, h };
}

/** Pixel quality inside the detected face oval. A padded rectangle includes
 * background, so a white wall used to reject a correctly exposed face and
 * sharp wallpaper could make a blurred face appear focused. Erode the mask
 * by two sampled pixels before computing the Laplacian to exclude silhouette
 * edges and pixels mixed with the background during downsampling. */
function faceImageQuality(gray: GrayPlane, imageFrame: Frame): { sharpness: number; clipped: number } {
  const bounds = clampRect(faceOvalBounds(imageFrame), gray.width, gray.height);
  const factor = Math.max(1, Math.ceil(Math.max(bounds.w, bounds.h) / 512));
  const roi = boxDownsample(cropGray(gray, bounds), 512);
  const polygon = FACE_OVAL.map(i => {
    const p = pt(imageFrame, i);
    return { x: (p.x - bounds.x) / factor, y: (imageFrame.imageHeight - p.y - bounds.y) / factor };
  });
  const { width: w, height: h, g } = roi;
  const mask = new Uint8Array(w * h);
  // Scanline rasterization is linear in the number of face pixels, with no
  // color/skin-tone threshold and no background assumptions.
  for (let y = 0; y < h; y++) {
    const intersections: number[] = [], scanY = y + .5;
    for (let i = 0; i < polygon.length; i++) {
      const a = polygon[i]!, b = polygon[(i + 1) % polygon.length]!;
      if ((a.y > scanY) !== (b.y > scanY)) intersections.push(a.x + (scanY - a.y) * (b.x - a.x) / (b.y - a.y));
    }
    intersections.sort((a, b) => a - b);
    for (let i = 0; i + 1 < intersections.length; i += 2) {
      const x0 = Math.max(0, Math.ceil(intersections[i]! - .5));
      const x1 = Math.min(w - 1, Math.floor(intersections[i + 1]! - .5));
      if (x1 >= x0) mask.fill(1, y * w + x0, y * w + x1 + 1);
    }
  }
  let count = 0, sum = 0, sum2 = 0, clipped = 0, lapSum = 0, lapSum2 = 0;
  for (let y = 2; y < h - 2; y++) for (let x = 2; x < w - 2; x++) {
    const i = y * w + x;
    let interior = !!mask[i];
    for (let dy = -2; dy <= 2 && interior; dy++) for (let dx = -2; dx <= 2; dx++) {
      if (!mask[i + dy * w + dx]) { interior = false; break; }
    }
    if (!interior) continue;
    const v = g[i]!;
    const lap = g[i - 1]! + g[i + 1]! + g[i - w]! + g[i + w]! - 4 * v;
    count++; sum += v; sum2 += v * v; lapSum += lap; lapSum2 += lap * lap;
    if (v <= 6 || v >= 249) clipped++;
  }
  if (count < 9) return { sharpness: 0, clipped: 1 };
  const variance = Math.max(0, sum2 / count - (sum / count) ** 2);
  const lapVariance = Math.max(0, lapSum2 / count - (lapSum / count) ** 2);
  return { sharpness: lapVariance / Math.max(variance, 1), clipped: clipped / count };
}

export function runGates(input: ScanInput, frame: Frame): GateReport {
  const blocking: Gate[] = [];
  const warnings: Gate[] = [];
  let confidenceMultiplier = 1;
  const regionConfidence: Partial<Record<FaceRegion, number>> = {};
  let jawEdgeSupport: number | null = null;

  if (!Number.isFinite(frame.scale) || frame.scale <= 1e-8) {
    return {
      pass: false,
      blocking: [gate("degenerate-landmarks", "block", "The eye reference points cannot define a measurement scale.", "Run detection again on a clear, front-facing photo.")],
      warnings, confidenceMultiplier: 0, regionConfidence, jawEdgeSupport,
    };
  }

  // 4.3 — face size
  if (frame.scale < MIN_IPD_PX) {
    blocking.push(
      gate(
        "face-too-small",
        "block",
        "Your face is too small in the frame.",
        "Move closer — your face should fill most of the frame.",
      ),
    );
  } else if (frame.scale < DEGRADE_IPD_PX) {
    warnings.push(
      gate(
        "face-small",
        "warn",
        "Your face is a little far away; measurements are less precise.",
        "Move closer for a sharper read.",
      ),
    );
    confidenceMultiplier *= 0.8;
  }
  if (Math.min(input.imageWidth, input.imageHeight) < MIN_FRAME_PX) {
    warnings.push(
      gate(
        "low-resolution",
        "warn",
        "The photo resolution is low.",
        "Use a higher-resolution camera or photo.",
      ),
    );
    confidenceMultiplier *= 0.85;
  }

  // 4.4 — pose
  const yaw = frame.yawDeg;
  if (frame.poseSource === "matrix" && yaw !== null) {
    const ay = Math.abs(yaw);
    if (ay > YAW_DEGRADE) {
      blocking.push(
        gate("yaw", "block", "Your head is turned too far to the side.", "Retake: face the camera straight on."),
      );
    } else if (ay > YAW_FULL) {
      warnings.push(
        gate("yaw-mild", "warn", "Your head is slightly turned; width measurements lose precision.", "Face the camera straight on."),
      );
      confidenceMultiplier *= 0.6;
    }
  } else {
    warnings.push(gate("pose-proxy", "warn", "Head alignment uses a landmark estimate; up/down tilt could not be checked.", "Use an eye-level, straight-on photo and inspect the landmark overlay."));
    const ap = Math.abs(frame.yawAsym);
    if (ap > YAW_PROXY_DEGRADE) {
      blocking.push(
        gate("yaw", "block", "The landmark alignment is too uneven for reliable front-view measurements.", "Face the camera straight on and check the detected points."),
      );
    } else if (ap > YAW_PROXY_FULL) {
      warnings.push(
        gate("yaw-mild", "warn", "Landmark alignment suggests a possible head turn; width measurements may lose precision.", "Face the camera straight on and check the detected points."),
      );
      confidenceMultiplier *= 0.6;
    }
  }
  const pitch = frame.pitchDeg;
  if (pitch !== null) {
    const apitch = Math.abs(pitch);
    if (apitch > PITCH_DEGRADE) {
      blocking.push(
        gate("pitch", "block", "Your head is tilted up or down too far.", "Hold the camera at eye level."),
      );
    } else if (apitch > PITCH_FULL) {
      warnings.push(
        gate("pitch-mild", "warn", "Your head is slightly tilted up or down.", "Hold the camera at eye level."),
      );
      confidenceMultiplier *= 0.6;
    }
  }
  const aroll = Math.abs(frame.rollDeg);
  if (aroll > ROLL_REFUSE) {
    blocking.push(gate("roll", "block", "Your head is tilted too far sideways.", "Keep your head level."));
  } else if (aroll > ROLL_DEGRADE) {
    warnings.push(gate("roll-mild", "warn", "Your head is slightly tilted.", "Keep your head level."));
    confidenceMultiplier *= 0.85;
  }

  // 4.7 — blink evidence is primary. A small opening ratio alone can be a
  // person's eye shape, so it cannot establish that the eye is closed.
  const bs = input.blendshapes;
  const blinkR = bs?.["eyeBlinkRight"];
  const blinkL = bs?.["eyeBlinkLeft"];
  const earR = eyeAspectRatio(frame, "right");
  const earL = eyeAspectRatio(frame, "left");
  const blendClosed =
    (blinkR !== undefined && blinkR > BLINK_REJECT) ||
    (blinkL !== undefined && blinkL > BLINK_REJECT);
  const unresolvedEye = !Number.isFinite(earR) || !Number.isFinite(earL) || earR <= 0 || earL <= 0;
  if (blendClosed) {
    blocking.push(
      gate("eyes-closed", "block", "One or both eyes look closed.", "Keep both eyes open and look at the camera."),
    );
  } else if (unresolvedEye) {
    blocking.push(gate("eyes-unresolved", "block", "The eyelid landmarks do not define a measurable eye opening.", "Use a sharp photo with visible eyes and check the detected points."));
  } else if ((blinkR === undefined || blinkL === undefined) && (earR < EAR_SMALL || earL < EAR_SMALL)) {
    warnings.push(gate("eye-opening-uncertain", "warn", "The eye opening is small and blink evidence is unavailable.", "Check that the eyelid points match the visible opening."));
    regionConfidence.eyes = Math.min(regionConfidence.eyes ?? 1, .6);
  } else if (Math.abs(earR - earL) > EAR_ASYM_WARN) {
    warnings.push(
        gate("eye-squint", "warn", "The measured eye openings differ; check for a blink or misplaced eyelid points.", "Relax your eyes and inspect the landmark overlay."),
    );
    regionConfidence.eyes = Math.min(regionConfidence.eyes ?? 1, 0.6);
  }

  // 4.8 — expression
  const jawOpen = bs?.["jawOpen"];
  const smile = Math.max(bs?.["mouthSmileLeft"] ?? 0, bs?.["mouthSmileRight"] ?? 0);
  if ((jawOpen !== undefined && jawOpen > JAW_OPEN_REJECT) || smile > SMILE_REJECT) {
    blocking.push(
      gate("expression", "block", "Your expression changes the geometry being measured.", "Relax your face — lips together, neutral expression."),
    );
  } else if ((jawOpen !== undefined && jawOpen > JAW_OPEN_WARN) || smile > SMILE_WARN) {
    warnings.push(
      gate("expression-mild", "warn", "A slight smile or open jaw shifts lower-face measurements.", "Relax your face for the most accurate read."),
    );
    confidenceMultiplier *= 0.8;
  }

  // 4.9a — reference-fit diagnostic. Shape difference is not proof of
  // occlusion, and cannot be used to punish natural facial variation.
  const residuals = regionalResiduals(frame);
  for (const region of Object.keys(residuals.byRegion) as FaceRegion[]) {
    const r = residuals.byRegion[region];
    if (r > RESIDUAL_REVIEW) {
      warnings.push(
        gate(
          `landmark-fit-${region}`,
          "warn",
          `The ${region} landmarks differ from the reference shape; this can be natural variation or detection error.`,
          "Inspect the points on the photo and correct them if they miss the visible features.",
        ),
      );
    }
  }

  // Image-dependent gates
  if (input.image) {
    // The photo itself is neither roll-corrected nor de-mirrored. Using
    // measurement coordinates here sampled background or the other cheek.
    const imageFrame = buildImageFrame(input, frame);
    const gray = toGray(input.image);
    const quality = faceImageQuality(gray, imageFrame);

    // 4.5 — blur
    const sharp = quality.sharpness;
    if (sharp < BLUR_REJECT) {
      blocking.push(gate("blur", "block", "The photo is too blurry to measure.", "Hold still and tap to focus, then retake."));
    } else if (sharp < BLUR_DEGRADE) {
      warnings.push(gate("blur-mild", "warn", "The photo is slightly soft; fine measurements lose precision.", "Hold still and tap to focus."));
      confidenceMultiplier *= 0.8;
    }

    // 4.6 — exposure
    const clipped = quality.clipped;
    if (clipped > CLIP_REJECT) {
      blocking.push(gate("exposure", "block", "The lighting is too dark or too bright.", "Face a window with soft, even light."));
    } else if (clipped > CLIP_DEGRADE) {
      warnings.push(gate("exposure-mild", "warn", "Parts of your face are over- or under-exposed.", "Face a window with soft, even light."));
      confidenceMultiplier *= 0.85;
    }

    // 4.9b — jaw edge support (never blocks; degrades jawline definition)
    jawEdgeSupport = computeJawEdgeSupport(imageFrame, gray);
    if (jawEdgeSupport < JAW_EDGE_MIN) {
      warnings.push(
        gate(
          "jaw-edge-weak",
          "warn",
          "We couldn't see your jawline clearly.",
          "Try pulling your collar down and separating your chin from your neck in the frame.",
        ),
      );
      regionConfidence.jaw = Math.min(regionConfidence.jaw ?? 1, 0.3);
    }
  } else warnings.push(gate("image-quality-unchecked", "warn", "Photo sharpness, lighting and visible jaw-edge evidence were not checked.", "Analyze the original photo to include image-quality checks."));

  return {
    pass: blocking.length === 0,
    blocking,
    warnings,
    confidenceMultiplier,
    regionConfidence,
    jawEdgeSupport,
  };
}

export type { GrayPlane };
