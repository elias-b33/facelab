"use client";

import { parseRatingModel, type RatingModel } from "@freeharmony/engine";

export const RATING_MODEL_KEY = "faceiq.rating-model.v1";
export const RATING_MODEL_EVENT = "faceiq:rating-model-changed";
export const MAX_RATING_MODEL_BYTES = 256_000;

export function readRatingModel(): { model: RatingModel | null; error: string | null } {
  if (typeof window === "undefined") return { model: null, error: null };
  try {
    const raw = localStorage.getItem(RATING_MODEL_KEY);
    if (!raw) return { model: null, error: null };
    if (new TextEncoder().encode(raw).byteLength > MAX_RATING_MODEL_BYTES) throw new Error("The saved model exceeds the supported size.");
    return { model: parseRatingModel(JSON.parse(raw.replace(/^\uFEFF/, ""))), error: null };
  } catch (error) {
    return { model: null, error: error instanceof Error ? error.message : "The saved model could not be read." };
  }
}

/** Import numerical data only; no URLs are fetched and no model code executes. */
export function saveRatingModel(raw: string): RatingModel {
  if (new TextEncoder().encode(raw).byteLength > MAX_RATING_MODEL_BYTES) throw new Error("Choose a model JSON smaller than 256 KB.");
  const model = parseRatingModel(JSON.parse(raw.replace(/^\uFEFF/, "")));
  try { localStorage.setItem(RATING_MODEL_KEY, JSON.stringify(model)); }
  catch { throw new Error("Browser storage is full. The previous model has been kept."); }
  window.dispatchEvent(new Event(RATING_MODEL_EVENT));
  return model;
}

export function removeRatingModel(): void {
  localStorage.removeItem(RATING_MODEL_KEY);
  window.dispatchEvent(new Event(RATING_MODEL_EVENT));
}
