"""E5-C80: fixed-budget inference-time component ablations (diagnostic, NOT retraining).
Run from D:\\DRTT\\E5_ClosedLoop; requires e5c75 and e5c76 scripts and frozen models.
Masking standardized-model inputs is distribution-shifting; results are sensitivity tests,
not causal component contributions. No test-based selection.
"""
import csv, json, hashlib, random, sys
from pathlib import Path
from collections import defaultdict
import numpy as np
ROOT=Path.cwd(); sys.path.insert(0,str(ROOT/'src'/'causal'))
import e5c5_overlapping_generator as gen
import e5c38_end_to_end as c38
import e5c58_compatible_gate_calibration as c58
import e5c75_calibrated_end_to_end as e75
import e5c76_signal_robust_detector as e76
OUT=ROOT/'results/causal/e5c80_component_ablation'
SEEDS=list(range(8001,8006)); ALERTS=1800; EDGE_BUDGET=400; W=90
# Mask components at both the reranker and gate. Indices follow E5-C56/C58 schemas.
RANK_MASK={'temporal':[0],'topology':[1,2,3,4],'signal':[7,8]}
GATE_MASK={'temporal':[5],'topology':[8,9,10,11],'signal':[6,7]}
ABLATIONS=['none','temporal','topology','signal','temporal+topology','temporal+signal','topology+signal','all']

def write(path,rows):
 if not rows:return
 cols=list(dict.fromkeys(k for r in rows for k in r))
 with path.open('w',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=cols);w.writeheader();w.writerows(rows)

def mask(X,components,mapping):
 Z=np.asarray(X,dtype=float).copy()
 for component in components:
  Z[:,mapping[component]]=0.
 return Z

class MaskedRanker:
 def __init__(self,model,components):self.model=model;self.components=components
 def decision_function(self,X):
  return self.model.decision_function(mask(X,self.components,RANK_MASK))

def summarise(rows):
 groups=defaultdict(list)
 for r in rows:groups[(r['detector'],r['variant'],r['ablation'])].append(r)
 ans=[]
 for (det,variant,ablation),rr in groups.items():
  tp=sum(x['tp'] for x in rr); pred=sum(x['predicted'] for x in rr); truth=sum(x['truth'] for x in rr)
  ans.append(dict(detector=det,variant=variant,ablation=ablation,seeds=len(rr),truth=truth,predicted=pred,tp=tp,fp=pred-tp,fn=truth-tp,
   alert_edge_recall=sum(x['alert_reachable_edges'] for x in rr)/truth if truth else 0,
   top1_correct=sum(x['top1_correct'] for x in rr),
   precision=tp/pred if pred else 0,recall=tp/truth if truth else 0,pooled_f1=e75.f1(tp,pred,truth)))
 return ans

def main():
 if OUT.exists():raise SystemExit(f'Refusing to overwrite: {OUT}')
 paths={'frozen':ROOT/'results/causal/e5c56/reranker_no_gap_rank.pkl',
        'gate':ROOT/'results/causal/e5c58/compatible_gate.pkl'}
 for p in paths.values():
  if not p.is_file():raise FileNotFoundError(p)
 frozen=e75.load(paths['frozen']);gate=e75.load(paths['gate'])
 if len(frozen['features'])!=10 or gate['features']!=c58.GATE_FEATURES:raise RuntimeError('Feature schema mismatch')
 detectors=e76.fit_models();rows=[]
 for seed in SEEDS:
  events,gt=gen.generate(seed,100,2400,10000)
  truth={(r['predecessor_event_id'],r['event_id']) for r in gt if r.get('predecessor_event_id')}
  labels={r['event_id']:int(r['malicious']) for r in gt}
  benign=[r['signal'] for r in events if labels[r['event_id']]==0]
  rng=random.Random(170000+seed)
  matched=[dict(r,signal=rng.choice(benign)) for r in events]
  for variant,ev in [('original',events),('matched_signal',matched)]:
   ids=[r['event_id'] for r in ev]; feats=c38.extract(ev,W)
   X=np.asarray([feats[i] for i in ids],dtype=float)
   for detname in ('OriginalOnly','NoSignal'):
    probs=detectors[detname].predict_proba(X)[:,1]
    chosen=set(i for i,_ in sorted(zip(ids,probs),key=lambda z:(-float(z[1]),z[0]))[:ALERTS])
    reachable=sum(pa in chosen and ch in chosen for pa,ch in truth)
    for ablation in ABLATIONS:
     components=[] if ablation=='none' else (list(RANK_MASK) if ablation=='all' else ablation.split('+'))
     wrapper={'model':MaskedRanker(frozen['model'],components),'features':frozen['features']}
     rr=c58.top_rows(ev,chosen,wrapper,W)
     if rr:
      gx=np.asarray([r['features'] for r in rr],dtype=float)
      gp=gate['model'].predict_proba(mask(gx,components,GATE_MASK))[:,1]
      pred=e75.top_budget(rr,gp,EDGE_BUDGET)
     else:pred=set()
     top1=sum((r['parent'],r['child']) in truth for r in rr)
     tp=len(pred&truth)
     rows.append(dict(seed=seed,detector=detname,variant=variant,ablation=ablation,alerts=ALERTS,
       edge_budget=EDGE_BUDGET,truth=len(truth),alert_reachable_edges=reachable,
       candidate_children=len(rr),top1_correct=top1,predicted=len(pred),tp=tp))
  print(f'test seed={seed} complete',flush=True)
 summary=summarise(rows)
 OUT.mkdir(parents=True,exist_ok=False)
 write(OUT/'test_seed_metrics.csv',rows)
 write(OUT/'test_summary.csv',summary)
 (OUT/'audit.json').write_text(json.dumps({'experiment':'E5-C80','test_seeds':SEEDS,'detector_train_seeds':e76.TRAIN,
  'alert_budget':ALERTS,'edge_budget':EDGE_BUDGET,'ablation_types':ABLATIONS,
  'ranker_mask_indices':RANK_MASK,'gate_mask_indices':GATE_MASK,
  'model_sha256':{k:hashlib.sha256(p.read_bytes()).hexdigest() for k,p in paths.items()},
  'limitations':['Synthetic-only','Inference-time zero masking, not feature-removal retraining','Gate best_score/margin and other derived features still reflect masked ranker scores; some proxy information remains','No test-based model selection','Matched signal resampling uses benign ground-truth labels','Frozen gate only','Fixed 1800 alerts and 400 predicted edges per seed']},indent=2),encoding='utf-8')
 print('\nE5-C80 SUMMARY')
 for r in summary:print(r)
 print('Saved:',OUT)
if __name__=='__main__':main()
