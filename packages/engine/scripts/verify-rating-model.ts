/** Check browser inference against a Python trainer's held-out predictions.
 * Usage: tsx scripts/verify-rating-model.ts features.jsonl model.json report.json
 */
import { readFileSync } from "node:fs";
import { parseAppearanceRatingModel, predictAppearanceRating } from "../src/rating/model";

const [featurePath, modelPath, reportPath] = process.argv.slice(2);
if (!featurePath || !modelPath || !reportPath) throw new Error("Provide feature JSONL, model JSON and report JSON paths.");
const model = parseAppearanceRatingModel(JSON.parse(readFileSync(modelPath, "utf8")));
const report = JSON.parse(readFileSync(reportPath, "utf8"));
const rows = new Map(readFileSync(featurePath, "utf8").replace(/^\uFEFF/, "").split(/\r?\n/).filter(Boolean).map(line => {
  const row = JSON.parse(line); return [row.id, row];
}));
let maximumDifference = 0;
for (const expected of report.testPredictions) {
  const row = rows.get(expected.id);
  if (!row) throw new Error(`Missing held-out feature row ${expected.id}`);
  const actual = predictAppearanceRating(row.features, model, { allowResearchModel: true });
  if (actual.status !== "predicted" || actual.score === null || !actual.interval) throw new Error(`Unexpected browser abstention: ${expected.id}: ${actual.reason}`);
  for (const [value, reference] of [[actual.score, expected.prediction], [actual.interval.lo, expected.interval[0]], [actual.interval.hi, expected.interval[1]]]) {
    const difference = Math.abs(value - reference);
    maximumDifference = Math.max(maximumDifference, difference);
    if (!Number.isFinite(difference) || difference > 1e-9) throw new Error(`Python/browser prediction mismatch: ${expected.id}`);
  }
}
for (const expected of report.testAbstentions) {
  const row = rows.get(expected.id);
  if (!row || predictAppearanceRating(row.features, model, { allowResearchModel: true }).status !== "unavailable") throw new Error(`Python/browser abstention mismatch: ${expected.id}`);
}
console.log(JSON.stringify({ modelId: model.id, predictionsMatched: report.testPredictions.length, abstentionsMatched: report.testAbstentions.length, maximumDifference }));
