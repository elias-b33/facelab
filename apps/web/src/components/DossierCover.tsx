import Link from "next/link";
import type { StoredScan } from "@/lib/store";
import { AnatomyPlate } from "./AnatomyPlate";

export function DossierCover({scan}:{scan:StoredScan}) {
  const score=scan.result.ok ? scan.result.overall : null;
  const strongest=[...scan.result.metrics].filter(m=>m.confidence >= .55).sort((a,b)=>b.score-a.score).slice(0,3);
  return <section className="dossier-cover">
    <div className="dossier-photographs"><figure>{scan.photo ? <img src={scan.photo} alt="Front portrait in your personal dossier"/> : <AnatomyPlate/>}<figcaption><span>01 / EN FACE</span><span>THE FRONT STUDY</span></figcaption></figure><figure>{scan.side?.photo ? <img src={scan.side.photo} alt="Side portrait in your personal dossier"/> : <Link className="add-profile-plate" href={"/scan/side?attach="+scan.id}><AnatomyPlate view="side"/><span>Add a profile photograph ↗</span></Link>}<figcaption><span>02 / IN PROFILE</span><span>{scan.side ? "THE SECOND PERSPECTIVE" : "COMPLETE YOUR STUDY"}</span></figcaption></figure></div>
    <div className="dossier-signature"><span className="folio">THE SIGNATURE REPORT</span><h2>Your signature.</h2><div className="signature-score"><span className="folio">ORIGINAL FRONT GEOMETRY INDEX</span><div>{score !== null ? <><strong>{score.toFixed(1)}</strong><span>/ 100</span></> : <strong className="score-pending">In review.</strong>}</div><p>{score !== null ? "FreeHarmony reference alignment across the original front metrics. It measures selected geometric proportions; it is not a calibrated looks rating." : "Your geometry draft is ready. An index is available when the front photograph passes its photo checks."}</p></div>{strongest.length > 0 && <div className="signature-strengths"><span className="folio">STRONGEST REFERENCE ALIGNMENTS</span>{strongest.map(metric=><div key={metric.key}><span>{metric.label}</span><strong>{metric.score.toFixed(1)}<small> / 100</small></strong></div>)}</div>}<Link className="text-button" href={"/results/"+scan.id+"/classic"}>The original score & methodology <span>↗</span></Link></div>
  </section>;
}
