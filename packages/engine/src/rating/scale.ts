/** A display convention, not a learned model or evidence of population rarity.
 * Freeze center/spread using HUMAN labels on the fit/reference set, never the
 * spread of model predictions (which would artificially amplify weak models).
 * This helper is intentionally not applied automatically to research ratings.
 */
export function strictDisplayRating(sourcePrediction: number, humanLabelCenter: number, humanLabelSpread: number, slope = 0.495): number {
  if (![sourcePrediction, humanLabelCenter, humanLabelSpread, slope].every(Number.isFinite) || humanLabelSpread <= 0 || slope <= 0) throw new Error("A finite prediction and positive human-label spread/slope are required.");
  const z = (sourcePrediction - humanLabelCenter) / humanLabelSpread;
  const logit = slope * z + Math.log(4 / 5);
  const probability = logit >= 0 ? 1 / (1 + Math.exp(-logit)) : Math.exp(logit) / (1 + Math.exp(logit));
  return Math.min(10 - 1e-12, Math.max(1 + 1e-12, 1 + 9 * probability));
}

/** One decimal, with 10 reserved as a theoretical ceiling even after rounding. */
export function formatStrictDisplayRating(value: number): string {
  if (!Number.isFinite(value) || value < 1 || value >= 10) throw new Error("Strict display ratings must be finite and in [1, 10).");
  return Math.min(9.9, Math.round(value * 10) / 10).toFixed(1);
}
