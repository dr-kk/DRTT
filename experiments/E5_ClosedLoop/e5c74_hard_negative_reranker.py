"""E5-C74: isolated hard-negative reranking on synthetic data.
Run from D:\\DRTT\\E5_ClosedLoop. No modification to frozen artifacts.
Training 7401-7410; validation 7411-7413; test 7414-7418.
Compares ranking on identical detector-selected candidates; gate deliberately not reused
because its calibration is specific to the frozen reranker.
"""
import csv, json, pickle, hashlib, sys
from pathlib import Path
from collections import defaultdict
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score

ROOT=Path.cwd();sys.path.insert(0,str(ROOT/'src'/'causal'))
import e5c5_overlapping_generator as gen
import e5c38_end_to_end as c38
import e5c15_reranker as base
OUT=ROOT/'results'/'causal'/'e5c74_hard_negative_reranker'
TRAIN=list(range(7401,7411));VAL=list(range(7411,7414));TEST=list(range(7414,7419));W=90

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):
 with p.open(newline='',encoding='utf-8-sig') as f:return list(csv.DictReader(f))
def dump(p,rows):
 if not rows:return
 keys=list(dict.fromkeys(k for r in rows for k in r))
 with p.open('w',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
def candidate_features(events,chosen):
 selected=sorted((r for r in events if r['event_id'] in chosen),key=lambda r:(int(r['timestamp']),r['event_id']))
 result={}
 for j,child in enumerate(selected):
  ct=int(child['timestamp']);pairs=[]
  for p in reversed(selected[:j]):
   gap=ct-int(p['timestamp'])
   if gap>W:break
   if gap<=0:continue
   if p['destination_host']==child['source_host'] or p['source_host']==child['source_host'] or p['destination_host']==child['destination_host']:
    pairs.append((p,gap))
  pairs.sort(key=lambda z:(z[1],z[0]['event_id']))
  if not pairs:continue
  arr=[]
  for i,(p,gap) in enumerate(pairs):
   arr.append([gap/90.,int(p['source_host']==child['source_host']),int(p['destination_host']==child['destination_host']),int(p['destination_host']==child['source_host']),int(p['source_host']==child['destination_host']),int(p['sensor']==child['sensor']),int(p['event_type']==child['event_type']),float(p['signal']),float(child['signal']),len(pairs)/50.,(i+1)/50.])
  result[child['event_id']]=([p['event_id'] for p,gap in pairs],np.asarray(arr,dtype=float)[:,:-1])
 return result

def data_for_seed(seed,detector,budget):
 events,truthrows=gen.generate(seed,100,2400,10000)
 truth={(r['predecessor_event_id'],r['event_id']) for r in truthrows if r['predecessor_event_id']}
 gt={c:p for p,c in truth}
 ids=[r['event_id'] for r in events];feats=c38.extract(events,W)
 probs=detector.predict_proba(np.asarray([feats[i] for i in ids]))[:,1]
 n=min(len(events),round(sum(float(r['signal'])>=.65 for r in events)*budget))
 chosen={i for i,_ in sorted(zip(ids,probs),key=lambda z:(-float(z[1]),z[0]))[:n]}
 return truth,gt,candidate_features(events,chosen)

def scores(model,x):return np.asarray(model.decision_function(x)).reshape(-1)
def rank(pids,x,model):
 s=scores(model,x)
 return sorted(range(len(pids)),key=lambda i:(-float(s[i]),i,pids[i]))

def evaluate(dataset,models,split):
 rows=[]
 for seed,(truth,gt,candidates) in dataset.items():
  reachable=[c for c,(pids,x) in candidates.items() if c in gt and gt[c] in pids]
  for name,model in models.items():
   top={1:0,3:0,5:0,10:0};edges=set()
   for child,(pids,x) in candidates.items():
    ix=rank(pids,x,model);edges.add((pids[ix[0]],child))
    if child in gt and gt[child] in pids:
     for k in top:top[k]+=int(gt[child] in {pids[i] for i in ix[:k]})
   tp=len(edges&truth);fp=len(edges)-tp
   rows.append(dict(split=split,seed=seed,model=name,truth=len(truth),reachable=len(reachable),predicted=len(edges),tp=tp,fp=fp,fn=len(truth)-tp,**{f'top{k}':v for k,v in top.items()}))
 return rows

def summary(rows):
 grouped=defaultdict(list)
 for r in rows:grouped[(r['split'],r['model'])].append(r)
 output=[]
 for (split,model),rr in grouped.items():
  truth=sum(r['truth'] for r in rr);reachable=sum(r['reachable'] for r in rr);tp=sum(r['tp'] for r in rr);fp=sum(r['fp'] for r in rr)
  output.append(dict(split=split,model=model,seeds=len(rr),truth=truth,reachable=reachable,tp=tp,fp=fp,
   pooled_f1=2*tp/(truth+tp+fp) if truth+tp+fp else 0,
   **{f'top{k}_accuracy':sum(r[f'top{k}'] for r in rr)/reachable if reachable else 0 for k in (1,3,5,10)}))
 return output

def main():
 if OUT.exists():raise SystemExit(f'Refusing to overwrite: {OUT}')
 paths={'old':ROOT/'results/causal/e5c15/reranker.pkl','new':ROOT/'results/causal/e5c56/reranker_no_gap_rank.pkl','gate':ROOT/'results/causal/e5c58/compatible_gate.pkl'}
 for k,p in paths.items():
  if not p.is_file():raise FileNotFoundError(f'{k}: {p}')
 with paths['old'].open('rb') as f:old=pickle.load(f)
 with paths['new'].open('rb') as f:frozen=pickle.load(f)
 with paths['gate'].open('rb') as f:gate=pickle.load(f)
 if old['features']!=base.FEATURES or frozen['features']!=base.FEATURES[:-1]:raise RuntimeError('Feature mismatch')
 if frozen.get('baseline_sha256')!=sha(paths['old']) or gate.get('e5c56_sha256')!=sha(paths['new']):raise RuntimeError('Frozen model lineage mismatch')
 tr=read(ROOT/'data/causal/e5c15_train/telemetry.csv')
 ty={r['event_id']:int(r['malicious']) for r in read(ROOT/'data/causal/e5c15_train/ground_truth.csv')}
 ff=c38.extract(tr,W);ids=[r['event_id'] for r in tr]
 detector=make_pipeline(StandardScaler(),LogisticRegression(max_iter=2000,random_state=0))
 detector.fit(np.asarray([ff[i] for i in ids]),np.asarray([ty[i] for i in ids]))
 datasets={}
 for split,seeds in [('train',TRAIN),('validation',VAL),('test',TEST)]:
  ds={}
  for seed in seeds:
   ds[seed]=data_for_seed(seed,detector,float(gate['budget']))
   print(f'{split} seed={seed} reachable={sum(c in ds[seed][1] and ds[seed][1][c] in pids for c,(pids,x) in ds[seed][2].items())}',flush=True)
  datasets[split]=ds
 # Include only children with a reachable true parent, consistent with frozen ranker training.
 X=[];y=[];groups=[];base_scores=[]
 for seed,(truth,gt,candidates) in datasets['train'].items():
  for child,(pids,x) in candidates.items():
   if child not in gt or gt[child] not in pids:continue
   yy=np.asarray([int(p==gt[child]) for p in pids]);X.extend(x);y.extend(yy)
   groups.extend([len(groups)]*0) # no-op; sample weights constructed per group below
   base_scores.extend(scores(frozen['model'],x))
 if not X:raise RuntimeError('No reachable training candidates')
 X=np.asarray(X);y=np.asarray(y,dtype=int);base_scores=np.asarray(base_scores)
 # Hard-negative weights: every positive has weight 1; each child gets a negative
 # weight of 1 only for its strongest wrong candidate under frozen model.
 # Reconstruct child boundaries deterministically (same iteration order).
 hard=np.ones(len(y));pos=0
 for seed,(truth,gt,candidates) in datasets['train'].items():
  for child,(pids,x) in candidates.items():
   if child not in gt or gt[child] not in pids:continue
   n=len(pids);ys=y[pos:pos+n];ss=base_scores[pos:pos+n]
   neg=np.flatnonzero(ys==0)
   if len(neg):
    hard[pos:pos+n]=0.2
    hard[pos+np.flatnonzero(ys==1)[0]]=1.0
    hard[pos+neg[np.argmax(ss[neg])]]=1.0
   pos+=n
 assert pos==len(y)
 # Select hard-negative multiplier exclusively by validation top1 accuracy.
 configs={'uniform':np.ones(len(y)),'hard_x2':hard*2,'hard_x5':hard*5}
 models={'Frozen':frozen['model']};validation=[]
 for name,weight in configs.items():
  m=LogisticRegression(max_iter=2500,class_weight='balanced',random_state=0)
  m.fit(X,y,sample_weight=weight)
  models[name]=m
  validation+=evaluate(datasets['validation'],{name:m},'validation')
 val_summary=summary(validation)
 winner=sorted(val_summary,key=lambda r:(-r['top1_accuracy'],-r['pooled_f1'],r['model']))[0]['model']
 print('Validation winner:',winner)
 test=evaluate(datasets['test'],{'Frozen':models['Frozen'],'SelectedHardNegative':models[winner]},'test')
 OUT.mkdir(parents=True,exist_ok=False)
 dump(OUT/'validation_seed_metrics.csv',validation);dump(OUT/'validation_summary.csv',val_summary)
 dump(OUT/'test_seed_metrics.csv',test);dump(OUT/'test_summary.csv',summary(test))
 with (OUT/'selected_model.pkl').open('wb') as f:pickle.dump({'model':models[winner],'features':base.FEATURES[:-1],'selection':winner,'validation_only':True},f)
 (OUT/'audit.json').write_text(json.dumps({'experiment':'E5-C74','train_seeds':TRAIN,'validation_seeds':VAL,'test_seeds':TEST,
  'model_sha256':{k:sha(p) for k,p in paths.items()},'selected_config':winner,'training_pairs':len(y),'training_positives':int(y.sum()),
  'evaluation':'ungated conditional ranking and ungated end-to-end edges; frozen gate NOT transferred to newly trained ranker',
  'limitation':'Synthetic only; matched-signal robustness not assessed; validation winner may be uniform'},indent=2))
 print('\nE5-C74 VALIDATION SUMMARY')
 for r in val_summary:print(r)
 print('\nE5-C74 TEST SUMMARY')
 for r in summary(test):print(r)
 print('Saved:',OUT)
if __name__=='__main__':main()
