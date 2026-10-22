import { describe, expect, it } from "vitest";
import { runGates } from "../src/gates/index";
import { FACE_OVAL } from "../src/landmarks/indices";
import { buildFrame } from "../src/normalize";
import type { ImageLike, ScanInput } from "../src/types";
import { canonicalInput } from "./helpers";

/** Independent scanline rendering: face pixels and background have separately
 * controlled exposure/texture. No image-processing implementation is reused. */
function photograph(input: ScanInput, background: number | "texture", texturedFace = true): ImageLike {
  const width = input.imageWidth, height = input.imageHeight;
  const data = new Uint8ClampedArray(width * height * 4);
  for (let y = 0; y < height; y++) for (let x = 0; x < width; x++) {
    const shade = background === "texture" ? 70 + ((x + y) % 7) * 20 : background;
    const offset = (y * width + x) * 4;
    data[offset] = data[offset + 1] = data[offset + 2] = shade;
    data[offset + 3] = 255;
  }
  const outline = FACE_OVAL.map(index => ({ x: input.landmarks[index]!.x * width, y: input.landmarks[index]!.y * height }));
  for (let row = 0; row < height; row++) {
    const y = row + 0.5;
    const intersections: number[] = [];
    for (let i = 0; i < outline.length; i++) {
      const a = outline[i]!, b = outline[(i + 1) % outline.length]!;
      if ((a.y > y) !== (b.y > y)) intersections.push(a.x + (y - a.y) * (b.x - a.x) / (b.y - a.y));
    }
    intersections.sort((a, b) => a - b);
    for (let pair = 0; pair + 1 < intersections.length; pair += 2) {
      const start = Math.max(0, Math.ceil(intersections[pair]! - 0.5));
      const stop = Math.min(width, Math.ceil(intersections[pair + 1]! - 0.5));
      for (let x = start; x < stop; x++) {
        const shade = texturedFace ? 70 + ((x + row) % 7) * 20 : 130;
        const offset = (row * width + x) * 4;
        data[offset] = data[offset + 1] = data[offset + 2] = shade;
      }
    }
  }
  return { width, height, data };
}

function blockingCodes(input: ScanInput) {
  return runGates(input, buildFrame(input).frame).blocking.map(gate => gate.code);
}

describe("photo checks use facial pixels", () => {
  it.each([0, 255])("does not reject an exposed sharp face because its background is %s", (background) => {
    const input = canonicalInput();
    input.image = photograph(input, background);
    expect(blockingCodes(input)).not.toContain("exposure");
    expect(blockingCodes(input)).not.toContain("blur");
  });

  it("does not let a sharp background conceal a textureless face", () => {
    const input = canonicalInput();
    input.image = photograph(input, "texture", false);
    expect(blockingCodes(input)).toContain("blur");
  });

  it("uses the original pixels after translated mirrored landmarks are normalized", () => {
    const input = canonicalInput({ mirrored: true });
    input.landmarks = input.landmarks.map(point => ({ ...point, x: 1 - (point.x + 0.06) }));
    input.image = photograph(input, 255);
    const codes = blockingCodes(input);
    expect(codes).not.toContain("exposure");
    expect(codes).not.toContain("blur");
  });
});
