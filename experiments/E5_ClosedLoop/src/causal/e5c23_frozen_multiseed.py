#!/usr/bin/env python3
"""E5-C23 frozen E5-C22 policies, five independent held-out synthetic seeds."""
import argparse, csv, importlib.util, json, math, statistics
from pathlib import Path
from types import SimpleNamespace

FROZEN = {
    "monitor_only": (0.55, 0.0),
    "isolate_only": (0.75, 0.0),
    "adaptive_no_forensics": (0.85, 0.0),
    "adaptive_with_forensics": (0.85, 0.30),
}
METRICS=("attack_completion_rate","false_isolation_rate","mean_action_cost",
         "mean_utility_loss","mean_preserved_evidence")

def load_module(path):
    spec=importlib.util.spec_from_file_location("e5c22_frozen_source",str(path))
    if not spec or not spec.loader: raise RuntimeError(f"Cannot import {path}")
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    return mod

def write_csv(path,rows):
    if not rows:return
    with path.open("w",newline="",encoding="utf8") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def ci95(xs):
    if len(xs)<2:return None
    return 1.96*statistics.stdev(xs)/math.sqrt(len(xs))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--source",default="src/causal/e5c22_forensic_feedback.py")
    ap.add_argument("--output",required=True)
    ap.add_argument("--seeds",nargs="+",type=int,default=[2060,2061,2062,2063,2064])
    ap.add_argument("--episodes",type=int,default=500)
    ap.add_argument("--max-steps",type=int,default=8)
    ap.add_argument("--benign-fraction",type=float,default=.5)
    ap.add_argument("--missing-rate",type=float,default=.1)
    ap.add_argument("--conflict-rate",type=float,default=.1)
    ap.add_argument("--tamper-rate",type=float,default=.05)
    a=ap.parse_args()
    if a.episodes<1 or a.max_steps<1 or len(a.seeds)<2 or len(set(a.seeds))!=len(a.seeds):
        ap.error("Require >=2 unique seeds and positive episode/step counts")
    if 2053 in a.seeds or 2054 in a.seeds:ap.error("Do not reuse E5-C22 calibration/test seeds")
    if any(not 0<=v<=1 for v in (a.benign_fraction,a.missing_rate,a.conflict_rate,a.tamper_rate)):
        ap.error("All rates must be between 0 and 1")
    src=Path(a.source)
    if not src.is_file():ap.error(f"Source file not found: {src}")
    m=load_module(src)
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    args=SimpleNamespace(max_steps=a.max_steps,missing_rate=a.missing_rate,
        conflict_rate=a.conflict_rate,tamper_rate=a.tamper_rate)
    allrows=[];perseed=[];pairrows=[]
    for seed in a.seeds:
        rows=[]
        for ep in range(a.episodes):
            cfg=m.scenario(seed,ep,a.benign_fraction)
            d={}
            for policy,(threshold,weight) in FROZEN.items():
                result,_=m.run_episode(cfg,policy,threshold,weight,args,False)
                rows.append(result);d[policy]=result
            nf=d["adaptive_no_forensics"];wf=d["adaptive_with_forensics"]
            pairrows.append(dict(seed=seed,episode=ep,benign=nf["benign"],
                delta_attack_completed=wf["attack_completed"]-nf["attack_completed"],
                delta_false_isolation=wf["false_isolation"]-nf["false_isolation"],
                delta_utility_loss=wf["utility_loss"]-nf["utility_loss"],
                delta_action_cost=wf["action_cost"]-nf["action_cost"],
                delta_preserved_evidence=wf["preserved_evidence"]-nf["preserved_evidence"]))
        allrows.extend(rows)
        for policy in FROZEN:
            sub=[r for r in rows if r["policy"]==policy]
            mal=[r for r in sub if not r["benign"]]
            ben=[r for r in sub if r["benign"]]
            if not mal or not ben:raise RuntimeError(f"Seed {seed}: missing malicious or benign class")
            perseed.append(dict(seed=seed,policy=policy,malicious_n=len(mal),benign_n=len(ben),
                attack_completion_rate=statistics.mean(r["attack_completed"] for r in mal),
                false_isolation_rate=statistics.mean(r["false_isolation"] for r in ben),
                mean_action_cost=statistics.mean(r["action_cost"] for r in sub),
                mean_utility_loss=statistics.mean(r["utility_loss"] for r in sub),
                mean_preserved_evidence=statistics.mean(r["preserved_evidence"] for r in sub)))
    write_csv(out/"per_seed.csv",perseed)
    write_csv(out/"paired_episodes.csv",pairrows)
    write_csv(out/"episodes.csv",allrows)
    comparison={}
    for metric in METRICS:
        diffs=[]
        for seed in a.seeds:
            d={r["policy"]:r for r in perseed if r["seed"]==seed}
            diffs.append(d["adaptive_with_forensics"][metric]-d["adaptive_no_forensics"][metric])
        comparison[metric]=dict(
            per_seed_differences=dict(zip(map(str,a.seeds),diffs)),
            mean_difference=statistics.mean(diffs),
            sd_across_seeds=statistics.stdev(diffs),
            approximate_95pct_ci_of_mean=[statistics.mean(diffs)-ci95(diffs),
                                          statistics.mean(diffs)+ci95(diffs)],
            seeds_with_improvement=sum(v<0 for v in diffs) if metric!="mean_preserved_evidence"
                                   else sum(v>0 for v in diffs))
    summary=dict(design="E5-C22 code reused with frozen calibration parameters; independent paired synthetic seeds",
        source=str(src),frozen_settings={k:{"threshold":v[0],"forensic_weight":v[1]} for k,v in FROZEN.items()},
        seeds=a.seeds,episodes_per_seed=a.episodes,comparisons=comparison,
        limitations=["No recalibration performed; thresholds hard-coded from E5-C22 seed 2053",
          "95% intervals are approximate normal intervals across only five seeds; descriptive, not definitive",
          "Synthetic temporal motif, not reconstructed causal chains",
          "Costs and utility weights are illustrative, not measured",
          "No PPO, no real network telemetry, no operational IRT"])
    (out/"summary.json").write_text(json.dumps(summary,indent=2),encoding="utf8")
    print("E5-C23 | FROZEN MULTI-SEED VALIDATION")
    print(json.dumps(summary,indent=2))
    print("Saved:",out.resolve())

if __name__=="__main__":main()
