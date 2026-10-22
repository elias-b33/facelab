/** Run the production geometry/quality engine over local, identity-audited rows.
 * Never derives human labels or identity IDs from filenames or geometry scores.
 * Use node --import tsx scripts/export-rating-features.ts --help.
 */
import { existsSync, mkdirSync, readFileSync, statSync, writeFileSync } from "node:fs";
import { dirname, isAbsolute, resolve } from "node:path";
import { pathToFileURL } from "node:url";
import { analyze, extractRatingFeatures, type Pt, type ScanInput } from "../src/index";

type Row = Record<string, unknown>;
const record = (value: unknown): value is Row => typeof value === "object" && value !== null && !Array.isArray(value);
const nonempty = (value: unknown): value is string => typeof value === "string" && value.trim().length > 0;

function readRows(path: string): Row[] {
  return readFileSync(path, "utf8").replace(/^\uFEFF/, "").split(/\r?\n/).filter(line => line.trim()).map((line, index) => {
    const value: unknown = JSON.parse(line);
    if (!record(value)) throw new Error(`${path}: row ${index + 1} is not an object.`);
    return value;
  });
}

/** Exported for integration tests; paths resolve relative to the source JSONL. */
export function convertRatingRows(rows: Row[], labelRows: Row[], sourceDirectory: string) {
  const labels = new Map<string, unknown>();
  for (const row of labelRows) {
    if (!nonempty(row.id) || labels.has(row.id)) throw new Error("Label rows need unique, nonempty photo IDs.");
    labels.set(row.id, row.rating);
  }
  const ids = new Set<string>();
  const subjects = new Set<string>();
  const exported = rows.map(row => {
    if (!nonempty(row.id) || ids.has(row.id)) throw new Error("Landmark rows need unique, nonempty photo IDs.");
    if (!nonempty(row.subject_id) || subjects.has(row.subject_id)) throw new Error(`${row.id}: supply one preselected photo per verified subject_id.`);
    if (row.identity_verified !== true || row.view !== "front") throw new Error(`${row.id}: verified person identity and view='front' are required.`);
    ids.add(row.id); subjects.add(row.subject_id);
    const rating = labels.size ? labels.get(row.id) : row.rating;
    if (!record(rating) || typeof rating.mean !== "number" || !Number.isFinite(rating.mean) ||
        !Number.isSafeInteger(rating.count) || (rating.count as number) < 5 || !Array.isArray(rating.scale) ||
        rating.scale.length !== 2 || !rating.scale.every(value => typeof value === "number" && Number.isFinite(value)) ||
        rating.scale[0] < 0 || rating.scale[1] > 10 || rating.scale[0] >= rating.scale[1] ||
        rating.mean < rating.scale[0] || rating.mean > rating.scale[1]) {
      throw new Error(`${row.id}: an exact joined human rating with mean, count>=5, and original scale is required.`);
    }
    const raw = Array.isArray(row.landmarks) ? row.landmarks : [];
    const landmarks = raw.map(point => Array.isArray(point)
      ? { x: point[0], y: point[1], z: point[2] }
      : point) as Pt[];
    const w = row.w as number, h = row.h as number;
    const input: ScanInput = {
      landmarks, imageWidth: w, imageHeight: h, mirrored: row.mirrored === true,
      transformationMatrix: row.matrix == null ? undefined : row.matrix as number[],
      blendshapes: row.blendshapes == null ? undefined : row.blendshapes as Record<string, number>,
      sex: "neutral", bandProfile: "calibrated",
    };
    let pixelsAvailable = false;
    if (nonempty(row.rgba_path)) {
      if (!Number.isSafeInteger(w) || !Number.isSafeInteger(h) || w <= 0 || h <= 0 || w * h > 16_777_216) throw new Error(`${row.id}: invalid or excessive pixel dimensions.`);
      const path = isAbsolute(row.rgba_path) ? row.rgba_path : resolve(sourceDirectory, row.rgba_path);
      if (statSync(path).size !== w * h * 4) throw new Error(`${row.id}: RGBA bytes do not match the stated dimensions.`);
      input.image = { width: w, height: h, data: new Uint8ClampedArray(readFileSync(path)) };
      pixelsAvailable = true;
    }
    const result = analyze(input);
    const featureRecord = extractRatingFeatures(result);
    const detectionIssue = nonempty(row.detection_issue) ? row.detection_issue : null;
    // No-image geometry is useful diagnostically, but cannot silently skip the
    // exposure/blur gates used in browser rating evaluation.
    const issues = [...featureRecord.issues, ...result.gates.blocking.map(gate => gate.message)];
    if (!pixelsAvailable) issues.push("Photo-quality pixels are absent; this row is excluded from supervised rating fitting.");
    if (detectionIssue) issues.push(detectionIssue);
    return { id: row.id, subject_id: row.subject_id, identity_verified: true, view: "front",
      ...featureRecord, quality_pass: featureRecord.quality_pass && pixelsAvailable && !detectionIssue,
      issues: [...new Set(issues)], rating };
  });
  const unused = [...labels.keys()].filter(id => !ids.has(id));
  if (unused.length) throw new Error(`${unused.length} labels have no matching landmark row. Check the photo ID join before training.`);
  return exported;
}

function main() {
  const args = process.argv.slice(2);
  if (args.includes("--help")) {
    console.log("Usage: node --import tsx scripts/export-rating-features.ts --input landmarks.jsonl [--labels labels.jsonl] --out features.jsonl\nRows require id, subject_id, identity_verified:true, view:'front', w, h, landmarks, optional matrix/blendshapes, rgba_path, and human rating (or exact label join). No photos or labels are downloaded. Failed detections remain excluded rows in the evaluation denominator.");
    return;
  }
  const options = new Map<string, string>();
  for (let i = 0; i < args.length; i += 2) {
    const key = args[i], value = args[i + 1];
    if (!key || !["--input", "--labels", "--out"].includes(key) || !value || value.startsWith("--") || options.has(key)) throw new Error("Invalid arguments. Use --help.");
    options.set(key, value);
  }
  const source = options.get("--input"), destination = options.get("--out");
  if (!source || !destination) throw new Error("--input and --out are required.");
  const inputPath = resolve(source), outputPath = resolve(destination);
  if (existsSync(outputPath)) throw new Error("The output already exists. Choose a new filename to preserve the previous export.");
  const labelPath = options.get("--labels");
  const labelRows = labelPath ? readRows(resolve(labelPath)) : [];
  if (labelPath && labelRows.length === 0) throw new Error("The explicitly supplied labels file is empty. No embedded-label fallback is used for an exact join.");
  const rows = convertRatingRows(readRows(inputPath), labelRows, dirname(inputPath));
  mkdirSync(dirname(outputPath), { recursive: true });
  writeFileSync(outputPath, rows.map(row => JSON.stringify(row)).join("\n") + "\n", { flag: "wx" });
  console.log(JSON.stringify({ total: rows.length, qualityAccepted: rows.filter(row => row.quality_pass).length,
    excluded: rows.filter(row => !row.quality_pass).length, output: outputPath }, null, 2));
}

if (process.argv[1] && import.meta.url === pathToFileURL(resolve(process.argv[1])).href) {
  try { main(); } catch (error) { console.error(error instanceof Error ? error.message : error); process.exitCode = 1; }
}
