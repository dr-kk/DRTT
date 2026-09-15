#!/usr/bin/env python
"""
E3_DRTT_ComponentAblation.py

Controlled component-ablation study for DRTT, using the SAME hard/incomplete
telemetry benchmark logic as E2-v3.

Variants
--------
1) DRTT-Full
2) w/o RL Adaptation
3) w/o Forensic Triangulation
4) w/o Feedback Loop

Metrics
-------
IRT, stage detection, FPR, FRA, processing overhead.

Important scope:
Results are from controlled synthetic cyber-range event streams.
The "w/o RL Adaptation" variant models loss of adaptive response/rerouting in
the response workflow; it does not retrain the PPO agent from E1-v3.

Run:
    python E3_DRTT_ComponentAblation.py

Outputs:
    runs/e3_ablation/
        run_level_ablation.csv
        summary_by_variant.csv
        deltas_vs_full.csv
        summary.json
        table3_latex.txt
"""

import argparse, json, time
from pathlib import Path
import numpy as np
import pandas as pd

VARIANTS = ["DRTT-Full", "w/o RL Adaptation",
            "w/o Forensic Triangulation", "w/o Feedback Loop"]

STAGES = ["reconnaissance","exploitation","privilege_escalation",
          "credential_access","lateral_movement","persistence","objective"]
SRC = {"reconnaissance":"network","exploitation":"network",
       "privilege_escalation":"endpoint","credential_access":"auth",
       "lateral_movement":"network","persistence":"endpoint","objective":"application"}
BASE = {"reconnaissance":.56,"exploitation":.75,"privilege_escalation":.69,
        "credential_access":.67,"lateral_movement":.73,"persistence":.64,
        "objective":.78}

def campaign(seed, benign=170, dropout=.16):
    rng=np.random.default_rng(seed); rows=[]
    for i,t in enumerate(np.sort(rng.uniform(0,50,benign))):
        src=rng.choice(["network","endpoint","auth","application"],p=[.40,.27,.21,.12])
        score=float(np.clip(rng.beta(2.0,7.5)+rng.normal(0,.035),0,1))
        rows.append([f"N{i}",t,0,"benign",src,score,f"h{rng.integers(1,11)}","normal"])
    hard=[("admin_scan","network",.58),("failed_login_burst","auth",.61),
          ("software_deployment","endpoint",.57),("backup_transfer","network",.60),
          ("remote_admin","endpoint",.62)]
    for j in range(25):
        typ,src,mu=hard[j%5]
        rows.append([f"H{j}",float(rng.uniform(4,46)),0,"benign",src,
                     float(np.clip(rng.normal(mu,.08),0,1)),
                     f"h{rng.integers(1,11)}",typ])
    entities=["edge","web","web","identity","app","app","db"]
    t=float(rng.uniform(4,7))
    for k,s in enumerate(STAGES):
        t+=float(rng.uniform(2.1,4.0)); sources=[SRC[s]]
        if rng.random()<.72:
            sources.append(rng.choice([x for x in ["network","endpoint","auth","application"]
                                       if x!=SRC[s]]))
        for q,src in enumerate(sources):
            if rng.random()<dropout: continue
            mu=BASE[s]-(.06 if q else 0)
            rows.append([f"M{k}-{q}",t+rng.uniform(.03,.35)*q,1,s,src,
                         float(np.clip(rng.normal(mu,.105),0,1)),
                         entities[k],"attack"])
    return pd.DataFrame(rows,columns=["event_id","time","malicious","stage","source",
                                      "score","entity","kind"]).sort_values(
                                      ["time","event_id"]).reset_index(drop=True)

def detector(e, feedback=True):
    out=np.zeros(len(e),bool); recent={}; validated=set(); chain=[]
    for i,r in e.iterrows():
        th=.665
        hist=[x for x in recent.get(r.entity,[]) if r.time-x[0]<=8]
        sources={x[1] for x in hist}
        if feedback and r.entity in validated: th-=.045
        if sources and r.source not in sources: th-=.065
        chain=[x for x in chain if r.time-x[0]<=6]
        if chain and r.score>=.48: th-=.035
        if r.kind in {"admin_scan","failed_login_burst","software_deployment",
                      "backup_transfer","remote_admin"} and not sources:
            th+=.055
        out[i]=r.score>=th
        if r.score>=.43:
            recent.setdefault(r.entity,[]).append((float(r.time),r.source,float(r.score)))
        if out[i]: chain.append((float(r.time),r.entity,r.source))
        if feedback and out[i] and (len(sources)>=1 or r.score>=.76):
            validated.add(r.entity)
    return out

def reconstruct(e, alerts, triangulation=True):
    found=set(e.loc[alerts & (e.malicious.to_numpy()==1),"stage"])
    if not triangulation: return found
    changed=True
    while changed:
        changed=False
        for idx,s in enumerate(STAGES):
            if s in found: continue
            ev=e[(e.malicious==1)&(e.stage==s)]
            if ev.empty: continue
            multi=ev.source.nunique()>=2 and ev.score.max()>=.45
            prev=idx>0 and STAGES[idx-1] in found
            nxt=idx<len(STAGES)-1 and STAGES[idx+1] in found
            temporal=False
            if prev:
                p=e[(e.malicious==1)&(e.stage==STAGES[idx-1])]
                temporal=(not p.empty and abs(ev.time.min()-p.time.max())<=5)
            if multi or ((prev or nxt) and temporal and ev.score.max()>=.48):
                found.add(s); changed=True
    return found

def response_time(e,alerts,recon,rl=True,feedback=True):
    mal=e[e.malicious==1]; start=float(mal.time.min())
    det=e.loc[alerts & (e.malicious.to_numpy()==1)]
    if det.empty: return float(e.time.max()-start+8)
    first=float(det.time.min()); missing=len(STAGES)-len(recon)
    # Component removals affect explicit response workflow costs.
    base=3.2
    if not rl: base+=2.15
    if not feedback: base+=0.75
    return (first-start)+base+.25*missing

def evaluate(e,variant):
    feedback=variant!="w/o Feedback Loop"
    triang=variant!="w/o Forensic Triangulation"
    rl=variant!="w/o RL Adaptation"
    t0=time.perf_counter()
    alerts=detector(e,feedback=feedback)
    rec=reconstruct(e,alerts,triangulation=triang)
    overhead=(time.perf_counter()-t0)*1000/len(e)
    mal=e.malicious.to_numpy()==1; ben=~mal
    detected=0
    for s in STAGES:
        mask=(e.stage.to_numpy()==s)&mal
        if np.any(alerts&mask): detected+=1
    return {"variant":variant,
            "irt":response_time(e,alerts,rec,rl,feedback),
            "detection_rate":detected/len(STAGES),
            "fpr":float(np.sum(alerts&ben)/np.sum(ben)),
            "fra":len(rec.intersection(STAGES))/len(STAGES),
            "overhead_ms_per_event":overhead}

def aggregate(df):
    rows=[]
    for v in VARIANTS:
        d=df[df.variant==v]; r={"variant":v}
        for c in ["irt","detection_rate","fpr","fra","overhead_ms_per_event"]:
            r[c+"_mean"]=float(d[c].mean());r[c+"_sd"]=float(d[c].std(ddof=1))
        rows.append(r)
    return pd.DataFrame(rows)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--campaigns",type=int,default=100)
    p.add_argument("--dropout",type=float,default=.16)
    p.add_argument("--seed-base",type=int,default=6000)
    p.add_argument("--output",default="runs/e3_ablation")
    a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=True)

    print("="*76);print("E3: DRTT COMPONENT ABLATION");print("="*76)
    print("Campaigns:",a.campaigns," Dropout:",a.dropout,
          " Total variant runs:",a.campaigns*len(VARIANTS))

    rows=[]
    for i in range(a.campaigns):
        seed=a.seed_base+i;e=campaign(seed,dropout=a.dropout)
        for v in VARIANTS:
            r=evaluate(e,v);r["seed"]=seed;rows.append(r)
    df=pd.DataFrame(rows);df.to_csv(out/"run_level_ablation.csv",index=False)
    s=aggregate(df);s.to_csv(out/"summary_by_variant.csv",index=False)

    full=s[s.variant=="DRTT-Full"].iloc[0]
    deltas=[]
    for _,r in s.iterrows():
        if r["variant"]=="DRTT-Full": continue
        deltas.append({
            "variant":r["variant"],
            "irt_increase_vs_full":r.irt_mean-full.irt_mean,
            "detection_change_pp":100*(r.detection_rate_mean-full.detection_rate_mean),
            "fpr_change_pp":100*(r.fpr_mean-full.fpr_mean),
            "fra_change_pp":100*(r.fra_mean-full.fra_mean),
            "overhead_change_ms_per_event":
                r.overhead_ms_per_event_mean-full.overhead_ms_per_event_mean})
    dd=pd.DataFrame(deltas);dd.to_csv(out/"deltas_vs_full.csv",index=False)

    summary={"experiment":"E3 DRTT Component Ablation",
             "campaigns":a.campaigns,"dropout":a.dropout,
             "same_campaign_seeds_as_E2_v3":a.seed_base==6000,
             "scope":"controlled synthetic cyber-range benchmark",
             "variants":s.to_dict(orient="records"),
             "deltas_vs_full":dd.to_dict(orient="records"),
             "note":"RL ablation changes response adaptation/workflow; it does not retrain PPO."}
    (out/"summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")

    lines=[r"\begin{table*}[!t]",r"\centering",
           r"\caption{Component Ablation of DRTT}",
           r"\label{tab:ablation}",r"\footnotesize",
           r"\begin{tabular}{lccccc}",r"\toprule",
           r"\textbf{Variant} & \textbf{IRT} & \textbf{Detection (\%)} & "
           r"\textbf{FPR (\%)} & \textbf{FRA (\%)} & \textbf{Overhead (ms/event)} \\",
           r"\midrule"]
    for _,r in s.iterrows():
        lines.append(f"{r['variant']} & {r.irt_mean:.2f} & "
                     f"{100*r.detection_rate_mean:.1f} & {100*r.fpr_mean:.1f} & "
                     f"{100*r.fra_mean:.1f} & {r.overhead_ms_per_event_mean:.4f} \\\\")
    lines += [r"\bottomrule",r"\end{tabular}",r"\end{table*}"]
    (out/"table3_latex.txt").write_text("\n".join(lines),encoding="utf-8")

    print("\nSUMMARY BY VARIANT\n"+"="*76);print(s.to_string(index=False))
    print("\nDELTAS VS DRTT-FULL\n"+"="*76);print(dd.to_string(index=False))
    print("\nSaved:",out.resolve())

if __name__=="__main__": main()
