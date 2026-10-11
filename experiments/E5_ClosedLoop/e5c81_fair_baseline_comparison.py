"""E5-C81: fair fixed-budget reconstruction baseline comparison.
Run from D:\\DRTT\\E5_ClosedLoop. Requires E5-C75 and E5-C76 scripts and saved models.
Frozen policies; independent seeds 8101..8105; no test-time tuning.
Baselines use the same detector-selected 1800 alerts, compatible candidate pairs,
and 400 output edges per seed. Heuristics are label-free, NOT learned SOTA baselines.
"""
import csv, json, random, hashlib, sys
from pathlib import Path
from collections import defaultdict, deque
import numpy as np
ROOT=Path.cwd();sys.path.insert(0,str(ROOT/'src'/'causal'))
import e5c5_overlapping_generator as gen
import e5c38_end_to_end as c38
import e5c58_compatible_gate_calibration as c58
import e5c75_calibrated_end_to_end as e75
import e5c76_signal_robust_detector as e76
OUT=ROOT/'results/causal/e5c81_fair_baseline_comparison'
SEEDS=list(range(8101,8106));ALERTS=1800;EDGES=400;W=90
METHODS=['Frozen','HardNegative','RecentParent','HostHandoff','TemporalHost','SeededRandom']

def dump(p,rows):
 if not rows:return
 cols=list(dict.fromkeys(k for r in rows for k in r))
 with p.open('w',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=cols);w.writeheader();w.writerows(rows)

def candidate_map(events,chosen):
 """Match E5-C58 parent compatibility: gap 1..90 and one of three host relations."""
 selected=sorted((r for r in events if r['event_id'] in chosen),key=lambda r:(int(r['timestamp']),r['event_id']))
 active=deque();candidates={}
 for child in selected:
  t=int(child['timestamp'])
  while active and t-int(active[0]['timestamp'])>W:active.popleft()
  cand=[]
  for parent in active:
   gap=t-int(parent['timestamp'])
   if gap<=0:continue
   a=int(parent['destination_host']==child['source_host'])
   b=int(parent['source_host']==child['source_host'])
   c=int(parent['destination_host']==child['destination_host'])
   if a or b or c:cand.append((parent['event_id'],gap,a,b,c))
  if cand:candidates[child['event_id']]=cand
  active.append(child)
 return candidates

def heuristics(cmap,probs,seed,variant):
 """Select one parent per child, then rank child edges by a label-free confidence score."""
 out={k:[] for k in METHODS[2:]}
 rng=random.Random(910000+seed+(0 if variant=='original' else 1))
 for child,candidates in cmap.items():
  # deterministic parent choice; ties broken by event ID
  recent=min(candidates,key=lambda x:(x[1],x[0]))
  handoff=min(candidates,key=lambda x:(-x[2],x[1],x[0]))
  combined=min(candidates,key=lambda x:(-(2*x[2]+x[3]+x[4]),x[1],x[0]))
  rand=candidates[rng.randrange(len(candidates))]
  for method,c in [('RecentParent',recent),('HostHandoff',handoff),('TemporalHost',combined),('SeededRandom',rand)]:
   parent,gap,a,b,d=c
   # Common child detector score ranks edge acceptance. Tiny deterministic tie-break
   # favors shorter gap and direct handoff; not tuned on any labels.
   confidence=float(probs[child])+1e-6*(W-gap)/W+1e-8*a
   out[method].append((confidence,parent,child))
 return out

def main():
 if OUT.exists():raise SystemExit(f'Refusing to overwrite: {OUT}')
 paths={'frozen':ROOT/'results/causal/e5c56/reranker_no_gap_rank.pkl',
        'hard':ROOT/'results/causal/e5c74_hard_negative_reranker/selected_model.pkl',
        'gate':ROOT/'results/causal/e5c58/compatible_gate.pkl',
        'new_gate':ROOT/'results/causal/e5c75_calibrated_end_to_end/selected_gate.pkl'}
 for k,p in paths.items():
  if not p.is_file():raise FileNotFoundError(f'{k}: {p}')
 models={k:e75.load(p) for k,p in paths.items()}
 if models['gate']['features']!=c58.GATE_FEATURES or models['new_gate']['features']!=c58.GATE_FEATURES:
  raise RuntimeError('Gate feature mismatch')
 detectors=e76.fit_models();detector=detectors['OriginalOnly']
 rows=[];diagnostics=[]
 for seed in SEEDS:
  ev,gt=gen.generate(seed,100,2400,10000)
  truth={(r['predecessor_event_id'],r['event_id']) for r in gt if r.get('predecessor_event_id')}
  labels={r['event_id']:int(r['malicious']) for r in gt}
  benign=[r['signal'] for r in ev if labels[r['event_id']]==0]
  rng=random.Random(170000+seed)
  variants={'original':ev,'matched_signal':[dict(r,signal=rng.choice(benign)) for r in ev]}
  for variant,events in variants.items():
   ids=[r['event_id'] for r in events];features=c38.extract(events,W)
   scores=detector.predict_proba(np.asarray([features[i] for i in ids],dtype=float))[:,1]
   ps=dict(zip(ids,map(float,scores)))
   chosen=set(sorted(ids,key=lambda i:(-ps[i],i))[:ALERTS])
   cmap=candidate_map(events,chosen)
   edge_coverage=sum(pa in chosen and ch in chosen for pa,ch in truth)
   candidates_coverage=sum(any(pa==p for p,*_ in cmap.get(ch,[])) for pa,ch in truth)
   preds={}
   for name,rkey,gkey in [('Frozen','frozen','gate'),('HardNegative','hard','new_gate')]:
    rr=c58.top_rows(events,chosen,models[rkey],W)
    if set(r['child'] for r in rr)!=set(cmap):
     raise RuntimeError(f'Candidate-child mismatch seed={seed} variant={variant} method={name}')
    if rr:
     X=np.asarray([r['features'] for r in rr],dtype=float)
     gp=models[gkey]['model'].predict_proba(X)[:,1]
     preds[name]=e75.top_budget(rr,gp,EDGES)
    else:preds[name]=set()
   for name,scored in heuristics(cmap,ps,seed,variant).items():
    ranked=sorted(scored,key=lambda z:(-z[0],z[2],z[1]))[:EDGES]
    preds[name]={(pa,ch) for _,pa,ch in ranked}
   for method in METHODS:
    pred=preds[method];tp=len(pred&truth)
    rows.append(dict(seed=seed,variant=variant,method=method,alerts=ALERTS,edge_budget=EDGES,
      truth=len(truth),alert_edge_reachable=edge_coverage,candidate_reachable=candidates_coverage,
      candidate_children=len(cmap),predicted=len(pred),tp=tp,fp=len(pred)-tp,fn=len(truth)-tp,
      precision=tp/len(pred) if pred else 0.,recall=tp/len(truth) if truth else 0.,
      f1=e75.f1(tp,len(pred),len(truth))))
   diagnostics.append(dict(seed=seed,variant=variant,truth=len(truth),alerts=ALERTS,
      alert_edge_reachable=edge_coverage,candidate_reachable=candidates_coverage,candidate_children=len(cmap)))
  print(f'test seed={seed} complete',flush=True)
 groups=defaultdict(list)
 for r in rows:groups[(r['variant'],r['method'])].append(r)
 summary=[]
 for (variant,method),group in groups.items():
  tp=sum(x['tp'] for x in group);pred=sum(x['predicted'] for x in group);truth=sum(x['truth'] for x in group)
  summary.append(dict(variant=variant,method=method,seeds=len(group),truth=truth,predicted=pred,
    tp=tp,fp=pred-tp,fn=truth-tp,precision=tp/pred if pred else 0.,recall=tp/truth if truth else 0.,
    pooled_f1=e75.f1(tp,pred,truth),alert_edge_recall=sum(x['alert_edge_reachable'] for x in group)/truth,
    candidate_recall=sum(x['candidate_reachable'] for x in group)/truth))
 OUT.mkdir(parents=True,exist_ok=False)
 dump(OUT/'test_seed_metrics.csv',rows);dump(OUT/'test_summary.csv',summary);dump(OUT/'candidate_diagnostics.csv',diagnostics)
 (OUT/'audit.json').write_text(json.dumps(dict(experiment='E5-C81',test_seeds=SEEDS,detector_train_seeds=e76.TRAIN,
  alert_budget=ALERTS,edge_budget=EDGES,detector='OriginalOnly',methods=METHODS,
  model_sha256={k:hashlib.sha256(p.read_bytes()).hexdigest() for k,p in paths.items()},
  limitations=['Synthetic-only','Baselines are deterministic heuristics, not state-of-the-art learned methods',
  'All methods share alerts and compatible candidate pairs and predict at most one parent per child',
  'Learned gates are trained/calibrated on different historical data than the heuristic acceptance scores',
  'Matched-signal perturbation is generated using benign ground-truth labels',
  'Fixed budgets were prespecified from E5-C77 validation; no test-time selection',
  'No statistical confidence intervals; reserved for E5-C82']),indent=2),encoding='utf-8')
 print('\nE5-C81 TEST SUMMARY')
 for r in summary:print(r)
 print('Saved:',OUT)
if __name__=='__main__':main()
