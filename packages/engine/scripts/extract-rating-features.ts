/** Local JSONL bridge: the research pipeline uses the SAME engine as the browser.
 * Usage: pnpm --filter @freeharmony/engine exec tsx scripts/extract-rating-features.ts input.jsonl output.jsonl
 * Optional rgbaPath points to a local raw RGBA buffer at scanInput dimensions.
 */
import { readFileSync, writeFileSync } from "node:fs";
import { resolve, dirname, isAbsolute } from "node:path";
import { analyze, ENGINE_VERSION, type ScanInput } from "../src/index";
import { extractRatingFeatures } from "../src/rating/features";

const [source, destination] = process.argv.slice(2);
if (!source || !destination) throw new Error("Provide input and output JSONL paths.");
if (resolve(source) === resolve(destination)) throw new Error("Output must differ from input.");
const rows = readFileSync(source, "utf8").replace(/^\uFEFF/, "").split(/\r?\n/).filter(line => line.trim());
let accepted = 0;
const result = rows.map(line => {
  const row = JSON.parse(line);
  if (row.detectionFailure && !row.scanInput) {
    const features = extractRatingFeatures({});
    features.quality.blockingReasons.push("Automatic detector did not find exactly one face.");
    return JSON.stringify({ id: row.id, subjectId: row.subjectId, dataset: row.dataset, source: row.source, label: row.label, detector: row.detector, engineVersion: ENGINE_VERSION, detectionFailure: row.detectionFailure, pixelGatesEvaluated: false, features });
  }
  const input = row.scanInput as ScanInput;
  if (!input || !Array.isArray(input.landmarks) || ![468, 478].includes(input.landmarks.length) || !input.landmarks.every(p => [p.x, p.y, p.z].every(Number.isFinite)) || !Number.isSafeInteger(input.imageWidth) || !Number.isSafeInteger(input.imageHeight) || input.imageWidth <= 0 || input.imageHeight <= 0) throw new Error(`Invalid scan input: ${row.id}`);
  if (row.rgbaPath) {
    const pixels = readFileSync(isAbsolute(row.rgbaPath) ? row.rgbaPath : resolve(dirname(source), row.rgbaPath));
    if (pixels.length !== input.imageWidth * input.imageHeight * 4) throw new Error(`RGBA dimensions mismatch: ${row.id}`);
    input.image = { data: new Uint8ClampedArray(pixels), width: input.imageWidth, height: input.imageHeight };
  }
  const scan = analyze({ ...input, sex: "neutral", bandProfile: "calibrated" });
  const features = extractRatingFeatures({ front: scan });
  if (!input.image) features.quality.warnings.push("Pixel quality gates were not evaluated in this extraction.");
  if (features.quality.frontUsable) accepted++;
  // No raw pixels, face mesh, path or sensitive photo content in the features file.
  return JSON.stringify({ id: row.id, subjectId: row.subjectId, dataset: row.dataset, source: row.source, label: row.label, detector: row.detector, engineVersion: ENGINE_VERSION, pixelGatesEvaluated: !!input.image, features });
});
writeFileSync(destination, result.join("\n") + "\n", "utf8");
console.log(JSON.stringify({ rows: result.length, usableFrontalPhotos: accepted, output: destination }));
