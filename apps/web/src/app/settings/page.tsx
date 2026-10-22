"use client";

import { useEffect, useRef, useState } from "react";
import Link from "next/link";
import { loadProfile, loadScans, nukeAllData, saveProfile, type Profile } from "@/lib/store";
import { downloadBackup, restoreBackup } from "@/lib/backup";
import { AiSettings } from "@/components/AiSettings";
import { RatingSettings } from "@/components/RatingSettings";
import { StudioArrow, StudioPageHeader } from "@/components/StudioPageHeader";

export default function SettingsPage() {
  const [profile,setProfile] = useState<Profile | null>(null);
  const [count,setCount] = useState(0);
  const [confirmDelete,setConfirmDelete] = useState(false);
  const [notice,setNotice] = useState("");
  const [error,setError] = useState("");
  const fileInput = useRef<HTMLInputElement>(null);
  useEffect(() => {setProfile(loadProfile());setCount(loadScans().length);},[]);
  const reportError = (reason: unknown) => {setNotice("");setError(reason instanceof Error ? reason.message : "The operation could not finish.");};
  return <main className="studio-page settings-page">
    <StudioPageHeader section="Your workspace" title={<>Your workspace<span className="title-stop">.</span></>} description="Your references, your backups and what is saved in this browser. A workspace under your control."/>
    {!profile ? <p className="studio-loading" role="status">Opening your workspace…</p> : <div className="workspace-layout">
      <aside className="workspace-index"><p className="studio-overline">IN THIS WORKSPACE</p><nav aria-label="Settings sections">{[["reference","01","Reference model"],["backups","02","Your backups"],["local-ai","03","Optional local AI"],["rating-model","04","Human-rating model"],["data-controls","05","Data controls"]].map(([id,n,label])=><a key={id} href={"#" + id}><span>{n}</span>{label}<StudioArrow/></a>)}</nav><div className="workspace-stamp"><span className="status-dot"/>SAVED ON THIS DEVICE<strong>{String(count).padStart(2,"0")}</strong><span>{count === 1 ? "personal study" : "personal studies"}</span></div></aside>
      <div className="workspace-sections">
        <section className="workspace-section" id="reference"><header><span>01</span><div><h2>Your reference model</h2><p>The comparison ranges used when you begin a new study.</p></div></header><div className="workspace-section-body"><div className="studio-segmented" role="group" aria-label="Reference model">{(["masculine","neutral","feminine"] as const).map(sex=><button key={sex} aria-pressed={profile.sex === sex} onClick={() => {const next={...profile,sex};try{saveProfile(next);setProfile(next);}catch(reason){reportError(reason);}}}>{sex}<span aria-hidden="true">{profile.sex === sex ? "✓" : ""}</span></button>)}</div><p className="studio-help">Existing reports keep the reference selected when they were created.</p><Link href="/welcome" className="studio-inline-link">Edit your study preferences<StudioArrow/></Link></div></section>
        <section className="workspace-section" id="backups"><header><span>02</span><div><h2>Backups & restore</h2><p>Export your photographs, measurements and landmark corrections together.</p></div></header><div className="workspace-section-body"><div className="backup-summary"><strong>{String(count).padStart(2,"0")}</strong><div><span>{count === 1 ? "study saved in this browser" : "studies saved in this browser"}</span><p>One private backup. Photos and measurements included.</p></div><span className="backup-format">JSON</span></div><div className="workspace-actions"><button className="primary-button" onClick={() => {try{downloadBackup();setError("");setNotice("Backup download started. Store it somewhere private.");}catch(reason){reportError(reason);}}}>Export backup <span>↓</span></button><button className="secondary-button" onClick={() => fileInput.current?.click()}>Restore a backup <span>↑</span></button></div><input ref={fileInput} type="file" accept=".json,application/json" className="hidden" aria-label="Restore a backup file" onChange={async event => {const file=event.target.files?.[0];event.target.value="";if(!file)return;try{if(file.size>20_000_000)throw new Error("Backup files must be smaller than 20 MB.");const added=restoreBackup(await file.text());setCount(loadScans().length);setProfile(loadProfile());setError("");setNotice("Restored " + added + (added === 1 ? " analysis." : " analyses.") + " Existing scans were kept.");}catch(reason){reportError(reason);}}}/><p className="studio-help">Restoring merges studies with your collection. Keep the same browser and site address to return to your history. Backups include your photos; save them somewhere private.</p></div></section>
        {notice && <p className="studio-notice" role="status">{notice}</p>}{error && <p className="studio-notice is-error" role="alert">{error}</p>}
        <section className="workspace-section" id="local-ai"><header><span>03</span><div><h2>Local report assistance</h2><p>Optional report assistance from an Ollama model on your own computer.</p></div></header><div className="workspace-section-body ai-settings-interior"><AiSettings/></div></section>
        <section className="workspace-section" id="rating-model"><header><span>04</span><div><h2>Human-rating model</h2><p>A separate prediction fitted to a named panel’s judgments.</p></div></header><div className="workspace-section-body"><RatingSettings/></div></section>
        <section className="workspace-section workspace-danger" id="data-controls"><header><span>05</span><div><h2>Local data controls</h2><p>Remove this app’s saved data from this browser. Exported backups stay where you saved them.</p></div></header><div className="workspace-section-body">{confirmDelete ? <div className="delete-confirmation"><p>This will remove all {count} saved studies and your settings from this browser.</p><div className="workspace-actions"><button onClick={() => {nukeAllData();window.location.href="/";}} className="secondary-button danger-button">Delete all local data</button><button onClick={() => setConfirmDelete(false)} className="secondary-button">Keep my collection</button></div></div> : <button onClick={() => setConfirmDelete(true)} className="studio-inline-link danger-link">Delete all local data<StudioArrow/></button>}</div></section>
      </div>
    </div>}
  </main>;
}
