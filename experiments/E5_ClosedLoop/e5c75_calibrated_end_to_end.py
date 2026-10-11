"""E5-C75: independent calibrated hard-negative ranker evaluation.
Run from D:\\DRTT\\E5_ClosedLoop. Does not overwrite previous results.
Gate fit: 7501-7505; gate C validation: 7506-7508; locked test: 7509-7513.
Same detector, alert budget, candidates, and per-seed output-edge budget for both rankers.
No ground truth accessed during inference; labels only for gate fitting/validation/evaluation.
"""
import csv, hashlib, json, pickle, random, sys
from pathlib import Path
from collections import defaultdict
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT=Path.cwd();sys.path.insert(0,str(ROOT/'src'/'causal'))
import e5c5_overlapping_generator as gen
import e5c38_end_to_end as c38
import e5c58_compatible_gate_calibration as c58
import e5c15_reranker as base

OUT=ROOT/'results/causal/e5c75_calibrated_end_to_end'
FIT=list(range(7501,7506));VAL=list(range(7506,7509));TEST=list(range(7509,7514))
W=90;DENSITY=16

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def load(p):
 if not p.is_file():raise FileNotFoundError(p)
 with p.open('rb') as f:return pickle.load(f) # Trusted locally generated models only.
def read(p):
 with p.open(newline='',encoding='utf-8-sig') as f:return list(csv.DictReader(f))
def dump(p,rows):
 if not rows:return
 keys=list(dict.fromkeys(k for r in rows for k in r))
 with p.open('w',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
def f1(tp,pred,truth):return 2*tp/(pred+truth) if pred+truth else 0.
def top_budget(rows,prob,budget):
 order=sorted(range(len(rows)),key=lambda i:(-float(prob[i]),rows[i]['child'],rows[i]['parent']))
 return {(rows[i]['parent'],rows[i]['child']) for i in order[:min(budget,len(order))]}
def frozen_budget(rows,gate):
 if not rows:return 0
 gp=gate['model'].predict_proba(np.asarray([r['features'] for r in rows]))[:,1]
 return int(sum(float(p)>=float(gate['cutoff']) for p in gp))
def prepare(seed,detector,frozen,new,gate):
 ev,ground=gen.generate(seed,100,2400,10000)
 truth={(r['predecessor_event_id'],r['event_id']) for r in ground if r['predecessor_event_id']}
 byid={r['event_id']:int(r['malicious']) for r in ground}
 benign=[r['signal'] for r in ev if byid[r['event_id']]==0]
 rng=random.Random(170000+seed)
 matched=[dict(r) for r in ev]
 for r in matched:r['signal']=rng.choice(benign)
 n=min(len(ev),round(sum(float(r['signal'])>=.65 for r in ev)*float(gate['budget'])))
 variants={}
 for variant,events in [('original',ev),('matched_signal',matched)]:
  ids=[r['event_id'] for r in events];ff=c38.extract(events,W)
  probs=detector.predict_proba(np.asarray([ff[i] for i in ids]))[:,1]
  chosen={i for i,_ in sorted(zip(ids,probs),key=lambda z:(-float(z[1]),z[0]))[:n]}
  fr=c58.top_rows(events,chosen,frozen,W)
  nr=c58.top_rows(events,chosen,new,W)
  if [r['child'] for r in fr]!=[r['child'] for r in nr]:raise RuntimeError('Candidate child mismatch')
  variants[variant]=(fr,nr)
 # Predeclare budget from ORIGINAL frozen global gate; same count for both models and variants.
 budget=frozen_budget(variants['original'][0],gate)
 return truth,variants,budget,n

def eval_one(ds,gate_new,gate_frozen,split,variants=('original','matched_signal')):
 rows=[]
 for seed,(truth,variants_map,budget,alerts) in ds.items():
  for variant in variants:
   fr,nr=variants_map[variant]
   for name,rr,gg in [('Frozen',fr,gate_frozen['model']),('HardNegative',nr,gate_new)]:
    if not rr:continue
    pr=gg.predict_proba(np.asarray([r['features'] for r in rr]))[:,1]
    pred=top_budget(rr,pr,budget)
    tp=len(pred&truth)
    rows.append(dict(split=split,seed=seed,variant=variant,model=name,truth=len(truth),alerts=alerts,
                     candidate_children=len(rr),budget=budget,predicted=len(pred),tp=tp,fp=len(pred)-tp,fn=len(truth)-tp,
                     precision=tp/len(pred) if pred else 0.,recall=tp/len(truth),f1=f1(tp,len(pred),len(truth))))
 return rows

def summarize(rows):
 groups=defaultdict(list)
 for r in rows:groups[(r['split'],r['variant'],r['model'])].append(r)
 out=[]
 for (split,variant,model),rr in groups.items():
  tp=sum(r['tp'] for r in rr);pred=sum(r['predicted'] for r in rr);truth=sum(r['truth'] for r in rr)
  out.append(dict(split=split,variant=variant,model=model,seeds=len(rr),truth=truth,predicted=pred,tp=tp,
                  fp=pred-tp,fn=truth-tp,pooled_precision=tp/pred if pred else 0.,pooled_recall=tp/truth if truth else 0.,pooled_f1=f1(tp,pred,truth)))
 return out

def main():
 if OUT.exists():raise SystemExit(f'Refusing to overwrite: {OUT}')
 paths={'old':ROOT/'results/causal/e5c15/reranker.pkl',
        'frozen':ROOT/'results/causal/e5c56/reranker_no_gap_rank.pkl',
        'gate':ROOT/'results/causal/e5c58/compatible_gate.pkl',
        'hard':ROOT/'results/causal/e5c74_hard_negative_reranker/selected_model.pkl'}
 for k,p in paths.items():
  if not p.is_file():raise FileNotFoundError(f'{k}: {p}')
 old=load(paths['old']);frozen=load(paths['frozen']);gate=load(paths['gate']);hard=load(paths['hard'])
 if old['features']!=base.FEATURES or frozen['features']!=base.FEATURES[:-1] or hard['features']!=base.FEATURES[:-1]:raise RuntimeError('Ranker feature mismatch')
 if gate['features']!=c58.GATE_FEATURES:raise RuntimeError('Gate feature mismatch')
 if frozen.get('baseline_sha256')!=sha(paths['old']) or gate.get('e5c56_sha256')!=sha(paths['frozen']):raise RuntimeError('Frozen lineage mismatch')
 audit74=json.loads((paths['hard'].parent/'audit.json').read_text(encoding='utf-8'))
 if hard.get('selection')!=audit74.get('selected_config') or audit74.get('test_seeds')!=list(range(7414,7419)):raise RuntimeError('E5-C74 selection provenance mismatch')
 tr=read(ROOT/'data/causal/e5c15_train/telemetry.csv')
 labels={r['event_id']:int(r['malicious']) for r in read(ROOT/'data/causal/e5c15_train/ground_truth.csv')}
 feats=c38.extract(tr,W);ids=[r['event_id'] for r in tr]
 detector=make_pipeline(StandardScaler(),LogisticRegression(max_iter=2000,random_state=0))
 detector.fit(np.asarray([feats[i] for i in ids]),np.asarray([labels[i] for i in ids]))
 ds={}
 for split,seeds in [('fit',FIT),('validation',VAL),('test',TEST)]:
  group={}
  for seed in seeds:
   group[seed]=prepare(seed,detector,frozen,hard,gate)
   print(f'{split} seed={seed} budget={group[seed][2]} candidates={len(group[seed][1]["original"][1])}',flush=True)
  ds[split]=group
 # Gate training uses only FIT seeds, and only ORIGINAL telemetry.
 X=[];y=[]
 for seed,(truth,variants,budget,alerts) in ds['fit'].items():
  for r in variants['original'][1]:
   X.append(r['features']);y.append(int((r['parent'],r['child']) in truth))
 X=np.asarray(X,dtype=float);y=np.asarray(y,dtype=int)
 if len(set(y.tolist()))!=2:raise RuntimeError('Gate fit labels contain only one class')
 # Select gate regularization on separate validation seeds; never on test.
 candidate_C=(0.01,0.1,1.,10.)
 gates={};validation=[]
 for C in candidate_C:
  m=LogisticRegression(max_iter=3000,class_weight='balanced',C=C,random_state=0)
  m.fit(X,y)
  gates[C]=m
  rr=eval_one(ds['validation'],m,gate,'validation',variants=('original',))
  ss=summarize(rr)
  hard_row=next(z for z in ss if z['model']=='HardNegative')
  validation.append({'C':C,**hard_row})
 winner=sorted(validation,key=lambda z:(-z['pooled_f1'],z['C']))[0]['C']
 print('Validation selected gate C:',winner)
 test=eval_one(ds['test'],gates[winner],gate,'test')
 summary=summarize(test)
 OUT.mkdir(parents=True,exist_ok=False)
 dump(OUT/'validation_gate_selection.csv',validation)
 dump(OUT/'test_seed_metrics.csv',test)
 dump(OUT/'test_summary.csv',summary)
 with (OUT/'selected_gate.pkl').open('wb') as f:pickle.dump({'model':gates[winner],'features':c58.GATE_FEATURES,'C':winner,'fit_seeds':FIT,'validation_seeds':VAL},f)
 (OUT/'audit.json').write_text(json.dumps({'experiment':'E5-C75','gate_fit_seeds':FIT,'gate_validation_seeds':VAL,'test_seeds':TEST,
  'selected_C':winner,'fit_gate_rows':len(y),'fit_gate_positive':int(y.sum()),'alert_budget_multiplier':float(gate['budget']),
  'budget_rule':'Per-seed frozen original Global cutoff edge count, fixed for both models and original/matched variants; label-free',
  'matched_rule':'Each event signal independently resampled from original benign marginal; original alert count retained',
  'model_sha256':{k:sha(p) for k,p in paths.items()},
  'limitations':['Synthetic generator only','Original frozen gate trained on earlier data; hard-negative gate trained on new fit seeds','Gate regularization chosen on validation only','Matched-signal counterfactual distribution shift','Equal edge budget does not imply equal calibration quality','Test seeds untouched by fitting or selection']},indent=2),encoding='utf-8')
 print('\nE5-C75 TEST SUMMARY')
 for r in summary:print(r)
 print('Saved:',OUT)
if __name__=='__main__':main()
