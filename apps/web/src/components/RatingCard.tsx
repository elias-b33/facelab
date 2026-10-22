"use client";

import { useEffect, useMemo, useState } from "react";
import Link from "next/link";
import { ENGINE_VERSION, extractRatingFeatures, predictRating, type RatingModel, type ScanResult } from "@freeharmony/engine";
import { readRatingModel, RATING_MODEL_EVENT, RATING_MODEL_KEY } from "@/lib/rating-model";
import { reanalyze } from "@/lib/scan";
import type { StoredScan } from "@/lib/store";

export function RatingCard({ scan }: { scan: StoredScan }) {
  const [model, setModel] = useState<RatingModel | null>(null);
  const [analysis, setAnalysis] = useState<ScanResult>(scan.result);
  const [busy, setBusy] = useState(false);
  const [modelError, setModelError] = useState("");
  const [analysisError, setAnalysisError] = useState("");
  useEffect(() => {
    const refresh = () => { const saved = readRatingModel(); setModel(saved.model); setModelError(saved.error ?? ""); };
    const storage = (event: StorageEvent) => { if (event.key === RATING_MODEL_KEY || event.key === null) refresh(); };
    refresh(); window.addEventListener(RATING_MODEL_EVENT, refresh); window.addEventListener("storage", storage);
    return () => { window.removeEventListener(RATING_MODEL_EVENT, refresh); window.removeEventListener("storage", storage); };
  }, []);
  useEffect(() => {
    let current = true;
    setAnalysis(scan.result);
    setAnalysisError("");
    if (scan.result.engineVersion === ENGINE_VERSION || !scan.input) { setBusy(false); return; }
    setBusy(true);
    // Recheck old stored landmarks/pixels locally. Never rewrite saved history.
    reanalyze(scan.input, scan.photo, scan.result.sex, scan.overrides)
      .then(result => { if (current) setAnalysis(result); })
      .catch(reason => { if (current) setAnalysisError(reason instanceof Error ? reason.message : "This photo could not be rechecked."); })
      .finally(() => { if (current) setBusy(false); });
    return () => { current = false; };
  }, [scan]);
  const prediction = useMemo(() => predictRating(analysis, model), [analysis, model]);
  const exportFeatures = () => {
    const record = { id: scan.id, subject_id: null, identity_verified: false, view: "front",
      ...extractRatingFeatures(analysis), rating: null,
      note: "Supply an independently verified subject ID and real human labels before training. No identity or rating has been inferred." };
    const url = URL.createObjectURL(new Blob([JSON.stringify(record) + "\n"], { type: "application/x-ndjson" }));
    const anchor = document.createElement("a"); anchor.href = url; anchor.download = `facelab-rating-features-${scan.id}.jsonl`; anchor.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  };
  return <section className="rating-panel card" aria-labelledby="rating-heading" aria-busy={busy}>
    <div className="section-heading"><div><p className="eyebrow">HUMAN-RATING MODEL</p><h2 id="rating-heading">Predicted panel rating</h2></div><Link className="text-button" href="/settings#rating-model">Model settings ↗</Link></div>
    {busy ? <p role="status">Rechecking this saved front photo with the current measurement engine…</p> : prediction.status === "predicted" ? <>
      <div className="rating-result"><strong>{prediction.value.toFixed(1)}</strong><div><span>Original {prediction.target.min}–{prediction.target.max} rating scale</span><p>{prediction.interval.low.toFixed(1)}–{prediction.interval.high.toFixed(1)} · {Math.round(prediction.interval.coverage * 100)}% nominal prediction interval</p></div></div>
      <p>{prediction.target.label}. Panel: {prediction.target.population}.</p>
      <dl className="rating-evidence"><div><dt>Held-out MAE</dt><dd>{prediction.evaluation.mae.toFixed(2)}</dd></div><div><dt>Mean baseline MAE</dt><dd>{prediction.evaluation.baselineMae.toFixed(2)}</dd></div><div><dt>Independent test people</dt><dd>{prediction.evaluation.testSubjects}</dd></div><div><dt>Test photos accepted</dt><dd>{Math.round(prediction.evaluation.testCoverage * 100)}%</dd></div></dl>
      <p className="reference-note">This interval concerns the panel mean for comparable accepted photos. It does not cover every viewer’s opinion or guarantee accuracy for this individual. The photo, lighting, pose and rating population matter.</p>
      <details className="metric-details"><summary>Model and prediction evidence</summary><p>{prediction.target.protocol}</p><p>Model {prediction.modelId}. Observed interval coverage in the held-out sample: {Math.round(prediction.evaluation.intervalCoverage * 100)}%. Evidence is reported by the imported model’s provider.</p><p>The rating uses the automatic front scan’s geometry. Named front/profile editor changes do not silently change these inputs. Profile estimates are excluded.</p></details>
    </> : <div className="rating-unavailable"><strong>{prediction.status === "unavailable" ? "Awaiting a trained model" : "Prediction withheld"}</strong>{prediction.messages.map(message => <p key={message}>{message}</p>)}</div>}
    {modelError && <p className="report-alert" role="alert">{modelError}</p>}
    {analysisError && <p className="report-alert" role="alert">{analysisError}</p>}
    <div className="rating-panel-footer"><p className="reference-note">The geometry alignment index and a predicted human rating answer different questions. This model keeps the rating scale on which it was trained.</p><button className="secondary-button" onClick={exportFeatures} disabled={busy}>Export front rating features ↓</button></div>
  </section>;
}
