"""E5-C60: frozen synthetic distribution-shift robustness audit.

Compares E5-C15+C31 against E5-C56+C58 on identical detector-selected alerts.
No model fitting except reconstruction of the E5-C15-trained detector; no tuning.
Shift axes are isolated: background count, campaign count, and observation
window duration. Duration is NOT a direct manipulation of attack inter-stage gaps.
Pickle inputs must be trusted local artifacts.
"""
import argparse,csv,hashlib,json,pickle,sys
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SCENARIOS={
 'reference':(100,2400,10000),
 'background_half':(100,2400,5000),
 'background_double':(100,2400,20000),
 'campaigns_half':(50,2400,10000),
 'campaigns_onehalf':(150,2400,10000),
 'duration_half':(100,1200,10000),
 'duration_double':(100,4800,10000),
}

def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()

def write_csv(path,rows):
 if not rows:return
 with path.open('w',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def scores_to_metrics(rows,probs,cutoff,truth):
 edges={(r['parent'],r['child']) for r,p in zip(rows,probs) if float(p)>=cutoff}
 tp=len(edges&truth);fp=len(edges-truth);fn=len(truth-edges)
 return {'tp':tp,'fp':fp,'fn':fn,'predicted':len(edges),
         'precision':tp/(tp+fp) if tp+fp else 0.,
         'recall':tp/(tp+fn) if tp+fn else 0.,
         'f1':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.}

def pooled(rows,prefix):
 tp=sum(int(r[prefix+'_tp']) for r in rows)
 fp=sum(int(r[prefix+'_fp']) for r in rows)
 fn=sum(int(r[prefix+'_fn']) for r in rows)
 return {'tp':tp,'fp':fp,'fn':fn,
         'precision':tp/(tp+fp) if tp+fp else 0.,
         'recall':tp/(tp+fn) if tp+fn else 0.,
         'f1':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.}

def main():
 p=argparse.ArgumentParser(description=__doc__)
 for name in ('ranker','retrained','gate','compatible-gate'):
  p.add_argument('--'+name,type=Path,required=True)
 p.add_argument('--train',type=Path,default=Path('data/causal/e5c15_train'))
 p.add_argument('--scripts',type=Path,default=Path('src/causal'))
 p.add_argument('--output',type=Path,required=True)
 p.add_argument('--seed-start',type=int,default=6000)
 p.add_argument('--num-seeds',type=int,default=5)
 a=p.parse_args()
 if a.num_seeds<2:p.error('--num-seeds must be >=2 for seed-level bootstrap')
 if a.output.exists():p.error('Output path exists; refusing to overwrite')
 for path in (a.ranker,a.retrained,a.gate,a.compatible_gate,a.train/'telemetry.csv',a.train/'ground_truth.csv'):
  if not path.is_file():p.error(f'Missing input: {path}')
 sys.path.insert(0,str(a.scripts.resolve()))
 import e5c15_reranker as base
 import e5c38_end_to_end as c38
 import e5c5_overlapping_generator as gen
 import e5c58_compatible_gate_calibration as c58
 with a.ranker.open('rb') as f:original=pickle.load(f)
 with a.retrained.open('rb') as f:modified=pickle.load(f)
 with a.gate.open('rb') as f:original_gate=pickle.load(f)
 with a.compatible_gate.open('rb') as f:modified_gate=pickle.load(f)
 if original['features']!=base.FEATURES or modified['features']!=base.FEATURES[:-1]:raise ValueError('Ranker feature mismatch')
 if original_gate['features']!=c58.GATE_FEATURES or modified_gate['features']!=c58.GATE_FEATURES:raise ValueError('Gate feature mismatch')
 if modified.get('baseline_sha256')!=sha(a.ranker):raise ValueError('Modified ranker lineage mismatch')
 if original_gate.get('e5c15_sha256')!=sha(a.ranker):raise ValueError('Original gate lineage mismatch')
 if modified_gate.get('e5c56_sha256')!=sha(a.retrained) or modified_gate.get('baseline_e5c15_sha256')!=sha(a.ranker):raise ValueError('Modified gate lineage mismatch')
 if float(original['threshold'])!=.65 or int(original['window'])!=90 or int(modified['window'])!=90 or int(modified_gate['window'])!=90:raise ValueError('Frozen settings mismatch')
 budget=float(modified_gate['budget'])
 if abs(budget-1.5)>1e-9:raise ValueError('Expected frozen budget 1.5x')
 train=c58.read(a.train/'telemetry.csv')
 labels={r['event_id']:int(r['malicious']) for r in c58.read(a.train/'ground_truth.csv')}
 ids=[r['event_id'] for r in train]
 x=c38.extract(train,90)
 detector=make_pipeline(StandardScaler(),LogisticRegression(max_iter=2000,class_weight=None,random_state=0))
 detector.fit(np.asarray([x[i] for i in ids]),np.asarray([labels[i] for i in ids]))
 per=[];summary=[]
 for scenario,(campaigns,duration,background) in SCENARIOS.items():
  sr=[]
  for seed in range(a.seed_start,a.seed_start+a.num_seeds):
   events,truthrows=gen.generate(seed,campaigns,duration,background)
   truth={(r['predecessor_event_id'],r['event_id']) for r in truthrows if r['predecessor_event_id']}
   event_ids=[r['event_id'] for r in events]
   feats=c38.extract(events,90)
   probs=detector.predict_proba(np.asarray([feats[i] for i in event_ids]))[:,1]
   base_n=sum(float(r['signal'])>=.65 for r in events)
   n=min(len(events),int(round(base_n*budget)))
   chosen={i for i,_ in sorted(zip(event_ids,probs),key=lambda z:(-float(z[1]),z[0]))[:n]}
   old_rows=c58.top_rows(events,chosen,original,90)
   new_rows=c58.top_rows(events,chosen,modified,90)
   if [r['child'] for r in old_rows]!=[r['child'] for r in new_rows]:raise RuntimeError(f'Candidate-child mismatch: {scenario} seed={seed}')
   old_probs=original_gate['model'].predict_proba(np.asarray([r['features'] for r in old_rows]))[:,1] if old_rows else []
   new_probs=modified_gate['model'].predict_proba(np.asarray([r['features'] for r in new_rows]))[:,1] if new_rows else []
   b=scores_to_metrics(old_rows,old_probs,float(original_gate['cutoff']),truth)
   q=scores_to_metrics(new_rows,new_probs,float(modified_gate['cutoff']),truth)
   row={'scenario':scenario,'seed':seed,'campaigns':campaigns,'duration':duration,'background':background,'events':len(events),'alerts':n,'truth_edges':len(truth),'candidate_children':len(old_rows)}
   row.update({'baseline_'+k:v for k,v in b.items()})
   row.update({'proposed_'+k:v for k,v in q.items()})
   row['delta_f1']=q['f1']-b['f1'];row['delta_fp']=q['fp']-b['fp']
   per.append(row);sr.append(row)
   print(f'{scenario} seed={seed}: truth={len(truth)} alerts={n} baseline TP/FP={b["tp"]}/{b["fp"]} F1={b["f1"]:.4f} proposed TP/FP={q["tp"]}/{q["fp"]} F1={q["f1"]:.4f} delta={row["delta_f1"]:+.4f}',flush=True)
  b=pooled(sr,'baseline');q=pooled(sr,'proposed')
  rng=np.random.default_rng(600060+list(SCENARIOS).index(scenario))
  ds=[]
  for _ in range(5000):
   sampled=[sr[int(i)] for i in rng.integers(0,len(sr),len(sr))]
   ds.append(pooled(sampled,'proposed')['f1']-pooled(sampled,'baseline')['f1'])
  lo,hi=np.percentile(ds,[2.5,97.5])
  r={'scenario':scenario,'campaigns':campaigns,'duration':duration,'background':background,'seeds':len(sr),'truth_edges':sum(z['truth_edges'] for z in sr),
     'baseline_tp':b['tp'],'baseline_fp':b['fp'],'baseline_precision':b['precision'],'baseline_recall':b['recall'],'baseline_f1':b['f1'],
     'proposed_tp':q['tp'],'proposed_fp':q['fp'],'proposed_precision':q['precision'],'proposed_recall':q['recall'],'proposed_f1':q['f1'],
     'delta_f1':q['f1']-b['f1'],'delta_fp':q['fp']-b['fp'],'f1_delta_ci_low':float(lo),'f1_delta_ci_high':float(hi),
     'seed_positive':sum(z['delta_f1']>0 for z in sr),'seed_zero':sum(z['delta_f1']==0 for z in sr),'seed_negative':sum(z['delta_f1']<0 for z in sr)}
  summary.append(r)
  print(f'SUMMARY {scenario}: baseline={b["f1"]:.4f} proposed={q["f1"]:.4f} delta={r["delta_f1"]:+.4f} CI=[{lo:+.4f},{hi:+.4f}] FP delta={r["delta_fp"]:+d}',flush=True)
 audit={'experiment':'E5-C60','status':'FROZEN SYNTHETIC DISTRIBUTION-SHIFT AUDIT','seed_range':[a.seed_start,a.seed_start+a.num_seeds-1],
        'alert_budget':budget,'scenarios':{k:{'campaigns':v[0],'duration':v[1],'background_events':v[2]} for k,v in SCENARIOS.items()},
        'model_sha256':{'baseline_ranker':sha(a.ranker),'proposed_ranker':sha(a.retrained),'original_gate':sha(a.gate),'compatible_gate':sha(a.compatible_gate)},
        'train_sha256':{'telemetry':sha(a.train/'telemetry.csv'),'ground_truth':sha(a.train/'ground_truth.csv')},
        'limitations':['All conditions use same synthetic generator family; not real-world domain shift','The same seed is reused across scenarios for paired comparisons, but event streams are regenerated','Duration changes observation-window density, not the 8-25 unit attack-stage gap distribution','No model fitting beyond rebuilding the frozen-training detector; no threshold selection','Seed-level bootstrap CIs are approximate with few seeds; seven scenario comparisons are exploratory']}
 a.output.mkdir(parents=True,exist_ok=False)
 write_csv(a.output/'per_seed.csv',per)
 write_csv(a.output/'scenario_summary.csv',summary)
 (a.output/'audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
 print('\nE5-C60 | DISTRIBUTION SHIFT SUMMARY')
 for r in summary:print(f'{r["scenario"]}: F1 {r["baseline_f1"]:.4f}->{r["proposed_f1"]:.4f} delta={r["delta_f1"]:+.4f} FP {r["baseline_fp"]}->{r["proposed_fp"]}')
 print('Saved:',a.output.resolve())
if __name__=='__main__':main()
