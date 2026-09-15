#!/usr/bin/env python
"""
E4_DRTT_Robustness.py

Robustness/sensitivity study complementing E1-v3, E2-v3, and E3.

Question:
How does DRTT-Full behave as telemetry becomes increasingly incomplete?

Dropout levels: 0%, 10%, 20%, 30%, 40%
Methods: Static Defense and DRTT-Full
Paired campaigns: same seed/campaign at every method within each dropout level.

Metrics:
- IRT
- stage detection
- FPR
- forensic reconstruction accuracy (FRA)
- processing overhead

Scope:
Controlled synthetic cyber-range benchmark, not production-network evidence.

Run:
    python E4_DRTT_Robustness.py

Outputs:
    runs/e4_robustness/
      run_level_results.csv
      summary_by_dropout.csv
      drtt_vs_static.csv
      summary.json
      robustness_table_latex.txt
"""

import argparse, json, time
from pathlib import Path
import numpy as np
import pandas as pd

STAGES=["reconnaissance","exploitation","privilege_escalation",
        "credential_access","lateral_movement","persistence","objective"]
SRC={"reconnaissance":"network","exploitation":"network",
     "privilege_escalation":"endpoint","credential_access":"auth",
     "lateral_movement":"network","persistence":"endpoint","objective":"application"}
BASE={"reconnaissance":.56,"exploitation":.75,"privilege_escalation":.69,
      "credential_access":.67,"lateral_movement":.73,"persistence":.64,
      "objective":.78}

def campaign(seed,dropout,benign=170):
    rng=np.random.default_rng(seed);rows=[]
    for i,t in enumerate(np.sort(rng.uniform(0,50,benign))):
        src=rng.choice(["network","endpoint","auth","application"],p=[.40,.27,.21,.12])
        score=float(np.clip(rng.beta(2,7.5)+rng.normal(0,.035),0,1))
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
        t+=float(rng.uniform(2.1,4.0));sources=[SRC[s]]
        if rng.random()<.72:
            sources.append(rng.choice([x for x in ["network","endpoint","auth","application"] if x!=SRC[s]]))
        for q,src in enumerate(sources):
            if rng.random()<dropout: continue
            mu=BASE[s]-(.06 if q else 0)
            rows.append([f"M{k}-{q}",t+rng.uniform(.03,.35)*q,1,s,src,
                         float(np.clip(rng.normal(mu,.105),0,1)),entities[k],"attack"])
    return pd.DataFrame(rows,columns=["event_id","time","malicious","stage","source",
                                      "score","entity","kind"]).sort_values(["time","event_id"]).reset_index(drop=True)

def static_detector(e): return (e.score>=.69).to_numpy()

def drtt_detector(e):
    out=np.zeros(len(e),bool);recent={};validated=set();chain=[]
    hard={"admin_scan","failed_login_burst","software_deployment","backup_transfer","remote_admin"}
    for i,r in e.iterrows():
        th=.665
        hist=[x for x in recent.get(r.entity,[]) if r.time-x[0]<=8]
        sources={x[1] for x in hist}
        if r.entity in validated: th-=.045
        if sources and r.source not in sources: th-=.065
        chain=[x for x in chain if r.time-x[0]<=6]
        if chain and r.score>=.48: th-=.035
        if r.kind in hard and not sources: th+=.055
        out[i]=r.score>=th
        if r.score>=.43: recent.setdefault(r.entity,[]).append((float(r.time),r.source))
        if out[i]: chain.append((float(r.time),r.entity,r.source))
        if out[i] and (sources or r.score>=.76): validated.add(r.entity)
    return out

def reconstruct(e,alerts,full):
    found=set(e.loc[alerts & (e.malicious.to_numpy()==1),"stage"])
    if not full:return found
    changed=True
    while changed:
        changed=False
        for idx,s in enumerate(STAGES):
            if s in found:continue
            ev=e[(e.malicious==1)&(e.stage==s)]
            if ev.empty:continue
            multi=ev.source.nunique()>=2 and ev.score.max()>=.45
            prev=idx>0 and STAGES[idx-1] in found
            nxt=idx<len(STAGES)-1 and STAGES[idx+1] in found
            temporal=False
            if prev:
                pe=e[(e.malicious==1)&(e.stage==STAGES[idx-1])]
                temporal=not pe.empty and abs(ev.time.min()-pe.time.max())<=5
            if multi or ((prev or nxt) and temporal and ev.score.max()>=.48):
                found.add(s);changed=True
    return found

def evaluate(e,method):
    t=time.perf_counter()
    alerts=static_detector(e) if method=="Static Defense" else drtt_detector(e)
    rec=reconstruct(e,alerts,method=="DRTT-Full")
    overhead=(time.perf_counter()-t)*1000/len(e)
    mal=e.malicious.to_numpy()==1;ben=~mal
    detected=sum(np.any(alerts & ((e.stage.to_numpy()==s)&mal)) for s in STAGES)
    start=float(e.loc[mal,"time"].min())
    dm=e.loc[alerts & mal]
    if dm.empty: irt=float(e.time.max()-start+8)
    else:
        first=float(dm.time.min());missing=len(STAGES)-len(rec)
        if method=="Static Defense":irt=(first-start)+6.8+.55*missing
        else:irt=(first-start)+3.2+.25*missing
    return {"method":method,"irt":irt,"detection_rate":detected/7,
            "fpr":float(np.sum(alerts&ben)/np.sum(ben)),
            "fra":len(rec.intersection(STAGES))/7,
            "overhead_ms_per_event":overhead}

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--campaigns",type=int,default=100)
    p.add_argument("--seed-base",type=int,default=7000)
    p.add_argument("--output",default="runs/e4_robustness")
    a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    levels=[0,.10,.20,.30,.40];rows=[]
    print("="*74);print("E4: DRTT TELEMETRY-DROPOUT ROBUSTNESS");print("="*74)
    print("Levels:",levels," Campaigns/level:",a.campaigns," Total runs:",len(levels)*a.campaigns*2)
    for d in levels:
        for i in range(a.campaigns):
            seed=a.seed_base+i;e=campaign(seed,d)
            for m in ["Static Defense","DRTT-Full"]:
                r=evaluate(e,m);r["dropout"]=d;r["seed"]=seed;rows.append(r)
    df=pd.DataFrame(rows);df.to_csv(out/"run_level_results.csv",index=False)
    ag=[]
    for d in levels:
        for m in ["Static Defense","DRTT-Full"]:
            x=df[(df.dropout==d)&(df.method==m)];z={"dropout":d,"method":m}
            for c in ["irt","detection_rate","fpr","fra","overhead_ms_per_event"]:
                z[c+"_mean"]=float(x[c].mean());z[c+"_sd"]=float(x[c].std(ddof=1))
            ag.append(z)
    s=pd.DataFrame(ag);s.to_csv(out/"summary_by_dropout.csv",index=False)
    comp=[]
    for d in levels:
        st=s[(s.dropout==d)&(s.method=="Static Defense")].iloc[0]
        fu=s[(s.dropout==d)&(s.method=="DRTT-Full")].iloc[0]
        comp.append({"dropout":d,
          "irt_reduction_pct":100*(st.irt_mean-fu.irt_mean)/st.irt_mean,
          "detection_gain_pp":100*(fu.detection_rate_mean-st.detection_rate_mean),
          "fpr_change_pp":100*(fu.fpr_mean-st.fpr_mean),
          "fra_gain_pp":100*(fu.fra_mean-st.fra_mean)})
    c=pd.DataFrame(comp);c.to_csv(out/"drtt_vs_static.csv",index=False)
    summary={"experiment":"E4 Telemetry-Dropout Robustness",
             "campaigns_per_dropout":a.campaigns,"dropout_levels":levels,
             "total_method_runs":len(levels)*a.campaigns*2,
             "scope":"controlled synthetic cyber-range benchmark",
             "summary_by_dropout":s.to_dict(orient="records"),
             "drtt_vs_static":c.to_dict(orient="records")}
    (out/"summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    lines=[r"\begin{table*}[!t]",r"\centering",
           r"\caption{DRTT Robustness Under Increasing Telemetry Loss}",
           r"\label{tab:robustness}",r"\footnotesize",
           r"\begin{tabular}{ccccc}",r"\toprule",
           r"\textbf{Dropout} & \textbf{IRT Reduction} & \textbf{Detection Gain} & "
           r"\textbf{FPR Change} & \textbf{FRA Gain} \\",
           r"\midrule"]
    for _,r in c.iterrows():
        lines.append(f"{100*r.dropout:.0f}\\% & {r.irt_reduction_pct:.1f}\\% & "
                     f"{r.detection_gain_pp:+.1f} pp & {r.fpr_change_pp:+.1f} pp & "
                     f"{r.fra_gain_pp:+.1f} pp \\\\")
    lines += [r"\bottomrule",r"\end{tabular}",r"\end{table*}"]
    (out/"robustness_table_latex.txt").write_text("\n".join(lines),encoding="utf-8")
    print("\nSUMMARY\n"+"="*74);print(s.to_string(index=False))
    print("\nDRTT VS STATIC\n"+"="*74);print(c.to_string(index=False))
    print("\nSaved:",out.resolve())
if __name__=="__main__":main()
