"use client";

import { useMemo, useState, type ChangeEvent } from "react";
import { extractRatingFeatures, parseAppearanceRatingModel, predictAppearanceRating, type AppearanceRatingModel, type HarmonySettings } from "@freeharmony/engine";
import type { StoredScan } from "@/lib/store";

/** Local-only model inspection. Research coefficients are never installed by default. */
export function AppearanceRating({ scan, settings }: { scan: StoredScan; settings: HarmonySettings }) {
  const [model, setModel] = useState<AppearanceRatingModel | null>(null);
  const [enabled, setEnabled] = useState(false);
  const [error, setError] = useState("");
  const features = useMemo(() => extractRatingFeatures({ front: scan.result, harmony: settings }), [scan.result, settings]);
  const prediction = useMemo(() => predictAppearanceRating(features, model, { allowResearchModel: enabled }), [features, model, enabled]);
  const measured = Object.values(features.values).filter(value => value !== null).length;

  async function loadModel(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0];
    event.target.value = "";
    if (!file) return;
    setError(""); setEnabled(false); setModel(null);
    try {
      if (file.size > 1_000_000) throw new Error("Choose a model JSON file smaller than 1 MB.");
      setModel(parseAppearanceRatingModel(JSON.parse(await file.text())));
    } catch (e) { setError(e instanceof Error ? e.message : "This model could not be read."); }
  }

  function exportFeatures() {
    const blob = new Blob([JSON.stringify({ format: "facelab-rating-features-v1", captureId: scan.id, features }, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url; link.download = `facelab-rating-features-${scan.id}.json`; link.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }

  return <section className="card report-tools" aria-labelledby="appearance-rating-title">
    <p className="eyebrow">APPEARANCE MODEL · RESEARCH</p>
    <h2 id="appearance-rating-title">A rating needs evidence.</h2>
    <p>The geometry index describes alignment with reference proportions. A looks estimate must learn independent human ratings and prove useful on new people.</p>
    <p><strong>{measured} reliable geometry inputs available.</strong> Missing or inferred anatomy stays excluded. No validated overall appearance model is installed.</p>
    <details className="metric-details">
      <summary>Inspect a local research model</summary>
      <p>Load the JSON artifact from a FaceLab training run. The model and measurements remain in this browser tab. The preview estimates the source panel’s mean rating of a frontal photograph, using its original rating scale.</p>
      <div className="export-buttons">
        <label className="secondary-button">Load research model<input className="sr-only" type="file" accept="application/json,.json" onChange={loadModel}/></label>
        <button className="secondary-button" onClick={exportFeatures}>Export measurement features</button>
      </div>
      {error && <p role="alert">{error}</p>}
      {model && <>
        <p><strong>{model.target.name}</strong> · {model.target.min}–{model.target.max} source scale · {model.training.dataset}</p>
        <p>Held-out mean absolute error: <strong>{model.evaluation.mae.toFixed(2)}</strong> points. Training-mean baseline: <strong>{model.evaluation.baselineMae.toFixed(2)}</strong>. Lower is better. Rank correlation: {model.evaluation.spearman?.toFixed(2) ?? "unavailable"}. The test partition contains {model.evaluation.testSubjects} people; errors are measured on accepted photos.</p>
        {model.evaluation.mae >= model.evaluation.baselineMae && <p><strong>This pilot did not outperform predicting the training average.</strong> Its ratings are not ready for general use.</p>}
        <p>Test acceptance: {(model.evaluation.acceptedFraction * 100).toFixed(0)}%. Observed interval coverage: {model.evaluation.intervalCoverage === null ? "unavailable" : `${(model.evaluation.intervalCoverage * 100).toFixed(0)}%`}. These measurements describe this small source study.</p>
        <label className="descriptive-toggle"><input type="checkbox" checked={enabled} onChange={event => setEnabled(event.target.checked)}/><span>Enable research prediction<small>This is an experimental photo estimate, not a validated personal looks rating.</small></span></label>
        <div role="status" aria-live="polite">
          {prediction.status === "predicted" && prediction.score !== null && prediction.interval && <p><strong>{prediction.score.toFixed(2)} on the {model.target.min}–{model.target.max} source scale</strong><br/>{(prediction.interval.nominalCoverage * 100).toFixed(0)}% nominal prediction interval: {prediction.interval.lo.toFixed(2)}–{prediction.interval.hi.toFixed(2)}.</p>}
          <p>{prediction.reason}</p>
        </div>
        <p className="reference-note">The initial pilot uses the original automatic frontal mesh. Edits to named atlas points and the profile view do not change that pilot’s inputs. Landmark confidence and review status do not measure prediction accuracy.</p>
        <button className="text-button" onClick={() => { setModel(null); setEnabled(false); }}>Unload model</button>
      </>}
      <p className="reference-note">Feature exports contain facial measurements and provenance, without photos. Keep them private. Exporting measurements does not create human training labels.</p>
    </details>
  </section>;
}
