import { describe, expect, it } from "vitest";
import { analyze } from "../src/index";
import { buildFrame } from "../src/normalize";
import { canonicalInput, projectCanonical, rotXYZ, toColMajor16 } from "./helpers";

describe("pose evidence is validated before it controls quality gates", () => {
  const notAffine = toColMajor16(rotXYZ(0, 0, 0));
  notAffine[3] = 0.5;
  const notRigid = toColMajor16(rotXYZ(0, 0, 0));
  notRigid[0] = 20;
  const notFinite = toColMajor16(rotXYZ(0, 0, 0));
  notFinite[5] = NaN;

  for (const [name, matrix] of [
    ["zero matrix", new Array<number>(16).fill(0)],
    ["non-affine matrix", notAffine],
    ["non-rigid matrix", notRigid],
    ["nonfinite matrix", notFinite],
  ] as const) {
    it(`falls back to geometric pose for a ${name}`, () => {
      const { frame } = buildFrame(canonicalInput({ transformationMatrix: [...matrix] }));
      expect(frame.poseSource).toBe("landmark-proxy");
      expect(frame.pitchDeg).toBeNull();
      expect(Number.isFinite(frame.yawDeg)).toBe(true);
    });
  }

  it("retains measured pitch when matrix yaw disagrees with an asymmetric landmark proxy", () => {
    // The proxy is weaker evidence than a valid rigid transform. Discarding
    // the matrix here used to silently remove the only pitch safety gate.
    const input = canonicalInput({
      landmarks: projectCanonical(rotXYZ(-8, 0, 0)),
      transformationMatrix: toColMajor16(rotXYZ(8, 14, 0)),
    });
    const { frame } = buildFrame(input);
    expect(frame.poseSource).toBe("matrix");
    expect(frame.pitchDeg).toBeCloseTo(14, 6);
    expect(analyze(input).gates.blocking.some(gate => gate.code === "pitch")).toBe(true);
  });

  it("does not interpret camera roll as a change in the landmark yaw proxy", () => {
    const straight = buildFrame(canonicalInput({ landmarks: projectCanonical(rotXYZ(8, 0, 0)), transformationMatrix: undefined })).frame;
    for (const roll of [-12, 12]) {
      const tilted = buildFrame(canonicalInput({ landmarks: projectCanonical(rotXYZ(8, 0, roll)), transformationMatrix: undefined })).frame;
      expect(tilted.yawAsym).toBeCloseTo(straight.yawAsym, 6);
      expect(tilted.yawDeg).toBeCloseTo(straight.yawDeg!, 6);
    }
  });
});
