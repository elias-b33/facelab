"use client";

import { useEffect, useRef, useState } from "react";
import { ratingModelReadiness, type RatingModel } from "@freeharmony/engine";
import { MAX_RATING_MODEL_BYTES, readRatingModel, removeRatingModel, saveRatingModel } from "@/lib/rating-model";

export function RatingSettings() {
  const [model, setModel] = useState<RatingModel | null>(null);
  const [notice, setNotice] = useState("");
  const [error, setError] = useState("");
  const input = useRef<HTMLInputElement>(null);
  useEffect(() => { const saved = readRatingModel(); setModel(saved.model); setError(saved.error ?? ""); }, []);
  const readiness = model ? ratingModelReadiness(model) : [];
  return <div className="rating-settings">
    <p>A trained model predicts the mean rating from its named human panel. Its original rating scale, measured test error and limitations stay visible in each dossier.</p>
    {model ? <div className="rating-model-summary">
      <strong>{model.provenance.datasetName}</strong>
      <p>{model.target.label} · {model.target.min}–{model.target.max} scale · {model.provenance.usage === "research" ? "Research use" : "Usage permission declared by model provider"}</p>
      <dl className="rating-evidence"><div><dt>Held-out people</dt><dd>{model.evaluation.testSubjects}</dd></div><div><dt>Mean absolute error</dt><dd>{model.evaluation.mae.toFixed(2)}</dd></div><div><dt>Mean baseline error</dt><dd>{model.evaluation.baselineMae.toFixed(2)}</dd></div><div><dt>Test photos accepted</dt><dd>{Math.round(model.evaluation.testCoverage * 100)}%</dd></div></dl>
      <p className="studio-help">Panel: {model.target.population}</p>
      <p className="studio-help">Model: {model.modelId}. These are the model provider’s recorded evaluation results; importing a file does not independently verify them.</p>
      {readiness.length > 0 && <div className="studio-notice is-error"><strong>Predictions withheld</strong>{readiness.map(issue => <p key={issue}>{issue}</p>)}</div>}
    </div> : <p className="studio-help">No trained rating model is installed. Automatic facial mapping and the geometry index are available now; a human-rating prediction needs fitted coefficients and independent evaluation.</p>}
    <div className="workspace-actions"><button className="primary-button" onClick={() => input.current?.click()}>Import model JSON ↑</button>{model && <button className="secondary-button" onClick={() => { try { removeRatingModel(); setModel(null); setError(""); setNotice("Rating model removed. Your studies are kept."); } catch { setError("The rating model could not be removed."); } }}>Remove model</button>}</div>
    <input ref={input} type="file" accept=".json,application/json" className="hidden" aria-label="Import a human-rating model" onChange={async event => {
      const file = event.target.files?.[0]; event.target.value = ""; if (!file) return;
      try {
        if (file.size > MAX_RATING_MODEL_BYTES) throw new Error("Choose a model JSON smaller than 256 KB.");
        const imported = saveRatingModel(await file.text()); setModel(imported); setError("");
        setNotice(ratingModelReadiness(imported).length ? "Model imported for inspection. Predictions are withheld until the listed issues are resolved." : "Model imported. Suitable front scans can now receive source-scale predictions.");
      } catch (reason) { setNotice(""); setError(reason instanceof Error ? reason.message : "The model could not be imported."); }
    }}/>
    {notice && <p className="studio-notice" role="status">{notice}</p>}{error && <p className="studio-notice is-error" role="alert">{error}</p>}
    <p className="studio-help">The model and predictions stay in this browser. Keep the original model JSON: photo backups do not include it. Synthetic test models can be inspected here but cannot rate people.</p>
  </div>;
}
